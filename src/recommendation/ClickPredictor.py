from src.recommendation.CustomModel import BertForSequenceClassificationAdapters, BertConfigAdapters
from transformers import AutoTokenizer
from typing import List
import torch
from rouge_score import rouge_scorer #pip install rouge-score
from huggingface_hub import hf_hub_download
import json
import os
import random
import numpy as np
import tempfile
import glob

class ClickPredictor():
  def __init__(self, huggingface_url : str, commit_hash : str = None, device : str = None):
    """
    load model from remote if not already present on disk. once the data is downloaded it is cached on your disk
    :param
      huggingface_url (str) : url pointing to the huggingface repository for example "josh-oo/news-classifier"
      commit_has (str) : the corresponding hash if you want to use a certain version for example "1b0922bb88f293e7d16920e7ef583d05933935a9"
      device (str) : if you want to execute it on a certain device for example "cpu" or "cuda"
    """
    config = BertConfigAdapters.from_pretrained(huggingface_url, revision=commit_hash)
    self.model = BertForSequenceClassificationAdapters.from_pretrained(huggingface_url, revision=commit_hash)
    self.tokenizer = AutoTokenizer.from_pretrained(huggingface_url, revision=commit_hash)

    #prepare cache file
    self.cache_dir = tempfile.TemporaryDirectory()

    #append personal embedding to to embedding matrix
    self.user_embedding_path = "personal_user_embedding.pt"

    personal_user_embedding = torch.ones(1, self.model.config.embedding_size).normal_(mean=0.0, std=self.model.config.initializer_range)
    if os.path.exists(self.user_embedding_path):
        personal_user_embedding = torch.load(self.user_embedding_path)
    else:
        torch.save(personal_user_embedding, self.user_embedding_path)

    with torch.no_grad():
        self.model.user_embeddings.weight.data = torch.concat([self.model.user_embeddings.weight.data, personal_user_embedding])

    #online learning hyperparameter
    learning_rate = 1e-2 / 4
    self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.1)
    self.positive_training_sample_path = "training_samples_positive.txt"
    self.negative_training_sample_path = "training_samples_negative.txt"

    self.device = device
    if self.device is not None:
      self.model.to(device)

  def _convert_tokens_to_words(self, tokens : List[str], values : torch.FloatTensor):
    """
    converts subword tokens to words and removes special tokens
    :param
      tokens : (List[str]) : the list of tokens
      values : torch.FloatTensor : the aggregated attentions for the corresponding tokens
    :return
      words : (List[str]) : a list containing all merged words
      values :  torch.FloatTensor: the modified attention map
    """
    all_words = []
    all_values = []
    current_word = ""
    current_value = 0
    for i,token in enumerate(tokens):
      if token in self.tokenizer.special_tokens_map.values():
        continue
      if not token.startswith("##") and len(current_word) > 0:
        all_words.append(current_word)
        all_values.append(current_value)
        current_word = ""
        current_value = 0
      if token.startswith("##"):
        token = token.replace("##","",1)
      current_word += token
      current_value += values[i].item()
    all_words.append(current_word)
    all_values.append(current_value)

    return all_words, torch.nn.functional.softmax(torch.tensor(all_values), dim=-1)

  def _extract_word_deviations(self, tokens : List[str], non_personal_attentions : torch.FloatTensor, personal_attentions : torch.FloatTensor):
    """
    aggregates the attention map (multi-head -> single head) and transforms subword tokens into full words
    :param
      tokens : (List[str]) : the list of tokens
      non_personal_attentions : torch.FloatTensor : the full (last) attention map for the non-personal prediction
      personal_attentions : torch.FloatTensor : the full (last) attention map for the personal prediction
    :return
      word_deviations : dict : a dict containing all words of the input headline and the deviation between personal and non-personal predictions
    """
    temperature = 1/12 #1/number_of_heads

    cls_attention_personal = torch.tensor(personal_attentions)
    cls_attention_non_personal = torch.tensor(non_personal_attentions)

    #aggregate with log summation (Adapted from: Attention Distillation: self-supervised vision transformer students need more guidance)
    personal_aggregated = torch.nn.functional.softmax(temperature * cls_attention_personal.log().sum(dim=-2),dim=-1)
    non_personal_aggregated = torch.nn.functional.softmax(temperature * cls_attention_non_personal.log().sum(dim=-2),dim=-1)

    words_personal, values_personal = self._convert_tokens_to_words(tokens, personal_aggregated)
    words_non_personal, values_non_personal = self._convert_tokens_to_words(tokens, non_personal_aggregated)

    deviation = (values_personal - values_non_personal) / values_non_personal

    all_deviations = {}
    for deviation, word in zip(deviation, words_personal):
      all_deviations[word] = deviation.item()

    return all_deviations

  def _get_score_and_attentions(self, headlines : List[str], user_id : str = "CUSTOM"):
    """
    calculates score and attention values for each headline or uses cache if available
    """
    all_cached_files = os.listdir(self.cache_dir.name)

    all_scores = [None] * len(headlines)
    all_attentions = [None] * len(headlines)

    remaining_headlines = []
    remaining_original_indices = []

    #check for each headline if it is already cached; if yes, load it from cache; else predict scores and attentions using the pytorch model
    for i, headline in enumerate(headlines):
      cache_file = str(user_id) + "#" + str(hash(headline)) + ".npz"
      if cache_file in all_cached_files:
        data = np.load(os.path.join(self.cache_dir.name, cache_file))
        all_scores[i] = data['score']
        all_attentions[i] = data['attentions']
      else:
        remaining_headlines.append(headline)
        remaining_original_indices.append(i)

    if len(remaining_headlines) > 0:
      inputs = self.tokenizer(remaining_headlines, return_tensors="pt", padding='longest').to(self.model.device)
      user_index = None
      if user_id == "CUSTOM":
        user_index = torch.tensor([len(self.model.user_embeddings.weight) -1], device=self.model.device).unsqueeze(dim=0)
      elif user_id is not None and user_id != "NONE":
        user_index = torch.tensor([user_id], device=self.model.device).unsqueeze(dim=0)

      self.model.eval()
      with torch.no_grad():
        outputs = self.model(**inputs, users=user_index, output_attentions=True)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        scores = probs[:,1].detach().numpy()
        attentions = outputs.attentions[-1]
        for i, score, attention in zip(remaining_original_indices,scores,attentions):
          all_scores[i] = score.item()
          #we are only interested in the cls token as it is used in our pooling step
          all_attentions[i] = attention[:,0].numpy()

          cache_file = str(user_id) + "#" + str(hash(headlines[i])) + ".npz"
          np.savez(os.path.join(self.cache_dir.name, cache_file), score=all_scores[i], attentions=all_attentions[i])

    return np.array(all_scores), all_attentions

  def calculate_scores(self, headlines : List[str], user_id : str = "CUSTOM", compare : bool = True):
    """
    calculate scores for a list of headlines for a given user
    :param
      headlines : (List[str]) : the list of headlines
      user_id (str) : either the user_id corresponding to the MIND dataset (for example) to calculate scores for historic users
                      or "CUSTOM" to calculate scores for the current new user
      compare : bool : if True -> compare results against non-personalized predictions (needed for wordclouds); if False -> calculate personal scores without comparison
    :return
      scores : (List[float]) : a score for each headline specifying a probability for a click event (1.0 = 100%)
      word_level_deviations (List[dict]): a list of dicts containing the headlines words and the deviation compared to the unpersonalized net
      personal_deviations (List[float]) : a list of floats indicating the deviation of the personal click probability compared to the non-personalized net
    """
    # assert user_id == "CUSTOM" or int(user_id) < 2500, "Given user id is not available" # todo complete check
    personal_scores, personal_attention = self._get_score_and_attentions(headlines,user_id)

    personal_deviations = None
    word_deviations = None

    if compare:
      non_personal_scores, non_personal_attentions = self._get_score_and_attentions(headlines,"NONE")
      personal_deviations = np.abs(np.array(personal_scores)-np.array(non_personal_scores))

      word_deviations = []
      inputs = self.tokenizer(headlines, return_tensors="pt", padding='longest')
      for i, headline in enumerate(inputs['input_ids']):
        current_tokens = self.tokenizer.convert_ids_to_tokens(headline)
        word_deviations.append(self._extract_word_deviations(current_tokens,non_personal_attentions[i], personal_attention[i]))

      #json_object = json.dumps({'personal_scores':personal_scores.tolist(), 'word_deviations':word_deviations, 'personal_deviations':personal_deviations.tolist()})
      #with open(os.path.join(self.cache_dir.name, cache_file), "w") as outfile:
      #  outfile.write(json_object)
      
    #cache highest and lowest rank for online learning
    current_highest_ranking = np.argmax(personal_scores)
    current_lowest_ranking = np.argmin(personal_scores)
    with open(os.path.join(self.cache_dir.name, "highest_ranking.txt"), "w") as highest:
        highest.write(headlines[current_highest_ranking])
    with open(os.path.join(self.cache_dir.name, "lowest_ranking.txt"), "w") as lowest:
        lowest.write(headlines[current_lowest_ranking])

    return personal_scores, word_deviations, personal_deviations

  def update_step(self, new_headline : str, new_label : int):
    """
    update the users embedding vector in this online learning step
    :param
      new_headline : (str) : the new headline
      new_label (int) : the label (in  {0,1}) 0 if the user did not like the new_headline, 1 if the user did like the new_headline
    """

    #add sample to file
    path = self.negative_training_sample_path if new_label == 0 else self.positive_training_sample_path
    with open(path, "a") as sample_file:
        sample_file.write(new_headline + "\n")

    #load all stored negative samples
    all_negative_samples = []
    if os.path.exists(self.negative_training_sample_path):
        with open(self.negative_training_sample_path) as f:
            all_negative_samples = [line.rstrip() for line in f]
            
    #load all stored positive samples
    all_positive_samples = []
    if os.path.exists(self.positive_training_sample_path):
        with open(self.positive_training_sample_path) as f:
            all_positive_samples = [line.rstrip() for line in f]
            
    if len(all_negative_samples) == 0 or len(all_positive_samples) == 0:
        if len(all_positive_samples) == 0:
            highest_ranking_path = os.path.join(self.cache_dir.name, "highest_ranking.txt")
            if os.path.exists(highest_ranking_path):
                with open(highest_ranking_path) as f:
                  all_positive_samples = [line.rstrip() for line in f]
        if len(all_negative_samples) == 0:
            lowest_ranking_path = os.path.join(self.cache_dir.name, "lowest_ranking.txt")
            if os.path.exists(lowest_ranking_path):
                with open(lowest_ranking_path) as f:
                  all_negative_samples = [line.rstrip() for line in f]
            
    #if there are still no samples, we have to skip this learning step
    if len(all_negative_samples) == 0 or len(all_positive_samples) == 0:
        return

    #sample k=4 negative samples
    k=4
    negative_samples = all_negative_samples[:k]
    if len(negative_samples) < k:
        #if there aer not enough negative samples we will oversample from the already existing negative samples
        negative_samples = random.choices(negative_samples, k=k)
    positive_sample = all_positive_samples[-1] #we will always use the last positive sample
    all_samples = negative_samples + [positive_sample]

    random_permutation = np.random.permutation(k + 1)
    all_samples = np.array(all_samples)[random_permutation]
    label = torch.tensor(np.where(random_permutation == 4)[0][0])

    #the personal user embedding is saved at the last embedding matrix index
    user_index = torch.tensor([len(self.model.user_embeddings.weight) -1])

    inputs = self.tokenizer(list(all_samples), return_tensors="pt", padding='longest')

    if self.device is not None:
        inputs = inputs.to(self.device)
        user_index = user_index.to(self.device)
        label = label.to(self.device)

    #freeze everything but the user_embeddings

    for param in self.model.parameters():
        param.requires_grad = False

    for param in self.model.user_embeddings.parameters():
        param.requires_grad = True


    #perform online learning update step
    self.model.train()

    output = self.model(**inputs, users=user_index.unsqueeze(dim=0), labels=label.unsqueeze(dim=0))
    loss = output.loss
    loss.backward()

    self.optimizer.step()
    self.optimizer.zero_grad()

    self.model.eval()

    #save updated personal user embedding
    torch.save(self.model.user_embeddings.weight[-1].unsqueeze(dim=0), self.user_embedding_path)

    #delete cached user files as they need to be recalculated
    all_cached_files = os.listdir(self.cache_dir.name)
    for cached_file in all_cached_files:
      if cached_file.startswith("CUSTOM"):
        os.remove(os.path.join(self.cache_dir.name, cached_file))


  def get_historic_user_embeddings(self):
    """
    :return: all embeddings in the shape (embedding_dim, total_number_of_users)
    """
    return self.model.user_embeddings.weight[:-1].detach().numpy()

  def get_personal_user_embedding(self):
    """
    :return: personlized embedding in the shape (embedding_dim, 1)
    """
    return self.model.user_embeddings.weight[-1].detach().numpy()

  def set_personal_user_embedding(self, user_index):
    """
    :param
      user_id : (str) : the mind user_id to initialize the useres embedding
    :return: personlized embedding in the shape (embedding_dim, 1)
    """
    #remove collected training samples
    for f in glob.glob("training_samples_*.txt"):
      os.remove(f)

    all_cached_files = os.listdir(self.cache_dir.name)
    for cached_file in all_cached_files:
      if cached_file.startswith("CUSTOM"):
        os.remove(os.path.join(self.cache_dir.name, cached_file))

    with torch.no_grad():
        self.model.user_embeddings.weight[-1] = self.model.user_embeddings.weight[user_index]

class RankingModule():
  def __init__(self, click_predictor : ClickPredictor):
    """
    initialize the ranking module
    :param
      the clickpredictor used to rank the headlines
    """
    self.click_predictor = click_predictor
    self.similarity_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

  def rank_headlines(self, ids : List[int], headlines : List[str], user_id : str = "CUSTOM", take_top_k : int = 10, exploration_ratio : float = 0.2):
    """
    get the k top ranked (distinct) articles including some exploration articles
    :param
      ids (List[int]) : list of ids to re-identify returned headlines and avoid duplicated "user-impressions"
      headlines (List[str]) : the candidate headlines to rank
      user_id (str) : the user
      take_top_k (int) : the number of articles to return
      exploration_rati (float) : the ratio of articles which are used for exploration (in the range of 0.0 and 1.0)
    :return: list of tuples containing the sorted candidate strings, the id and the score for example: [("Lorem ipsum ...", 2, 0.78)]
    """
    #TODO double check ranking algorithm
    assert len(headlines) > take_top_k
    assert exploration_ratio >= 0.0 and exploration_ratio <= 1.0
    assert len(headlines) == len(ids)

    scores, _, _ = self.click_predictor.calculate_scores(headlines, user_id, compare=False) #compare is set to True to trigger caching

    headlines_sorted = [x for _, x in sorted(zip(scores, headlines), reverse=True)]
    ids_sorted =[x for _, x in sorted(zip(scores, ids), reverse=True)]
    scores_sorted = sorted(scores, reverse=True)
    headlines_ids_sorted = list(zip(headlines_sorted, ids_sorted, scores_sorted))

    k_best = int(take_top_k * (1.0 - exploration_ratio))

    similarity_threshold = 0.5

    selected_headlines = []
    while len(selected_headlines) < k_best and len(headlines_sorted) > 0:
      candidate, index, score = headlines_ids_sorted.pop(0)

      #calculate similarity to already existing candidates
      similar_headline_already_selected = False
      for item, _, _ in selected_headlines:
        scores = self.similarity_scorer.score(candidate, item)['rougeL'].fmeasure
        if scores > similarity_threshold:
          similar_headline_already_selected = True
          break

      if similar_headline_already_selected == False:
        selected_headlines.append((candidate, index, score))

    #reverse headlines for low ranked articles
    headlines_ids_sorted = reversed(headlines_ids_sorted)

    #fill the remaining space with exploration headlines
    while len(selected_headlines) < take_top_k:
      try:
        candidate, index, score = headlines_ids_sorted.pop(0)
      except Exception:
        break

      #calculate similarity to already existing candidates
      similar_headline_already_selected = False
      for item, _, _ in selected_headlines:
        scores = self.similarity_scorer.score(candidate, item)['rougeL'].fmeasure
        if scores > similarity_threshold:
          similar_headline_already_selected = True
          break

      if similar_headline_already_selected == False:
        selected_headlines.append((candidate, index, score))

    return selected_headlines
