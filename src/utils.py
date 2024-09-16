import glob
import os
import numpy as np
import pandas as pd
import streamlit as st
from wordcloud import WordCloud
from collections import Counter
from wordcloud import STOPWORDS
from wordcloud import get_single_color_func


def remove_old_files():
    """
    If its the first run or if the session state needs to be cleaned,
    all files generated for the recommender model are deleted
    """
    if 'clean' not in st.session_state or st.session_state['clean'] is False:
        try:
            os.remove('personal_user_embedding.pt')
            for f in glob.glob("training_samples_*.txt"):
                os.remove(f)
        except OSError:
            pass
        st.session_state['clean'] = True


@st.cache_data
def load_data(path):
    return np.load(path)


@st.cache_data
def load_headlines():
    headline_path = st.session_state.config['HeadlinePath']
    return pd.read_csv(headline_path, header=None, sep='\t').tail(int(st.session_state.config['NoHeadlines'])) \
        .reset_index(drop=True)


def generate_wordcloud(word_dict):
    color_function = get_single_color_func('#3070B3')
    return WordCloud(scale=3, contour_width=0, color_func=color_function, width=200, height=250,
                     background_color="rgba(255, 255, 255, 0)", mode="RGBA") \
        .generate_from_frequencies(word_dict)


def generate_header():
    l_small, l_right = st.columns([1, 5])
    l_small.image('media/logo_dark.png', use_column_width='always')
    l_right.title('Balanced Article Discovery')
    l_right.title('through Playful User Nudging')


def set_session_state(emergency_user):
    if 'cold_start' not in st.session_state:
        st.session_state['cold_start'] = emergency_user
    if 'user' not in st.session_state:
        st.session_state['user'] = st.session_state['cold_start']
    if 'article_mask' not in st.session_state:
        st.session_state['article_mask'] = np.array(
            [True] * (int(
                st.session_state.config['NoHeadlines'])))


def reset_session_state(cold_start_user):
    st.session_state['cold_start'] = cold_start_user
    st.session_state['user'] = st.session_state['cold_start']
    st.session_state['article_mask'] = np.array(
        [True] * (int(
            st.session_state.config['NoHeadlines'])))


def extract_unread(headlines):
    """
    Returns only the headlines, which have not yet been marked as read in the session stat
    """
    unread_headlines_ind = np.nonzero(st.session_state.article_mask)[0]
    unread_headlines = list(headlines.loc[:, 2][st.session_state.article_mask])
    return unread_headlines_ind, unread_headlines


def get_wordcloud_from_attention(scores, word_deviations, personal_deviations, mode='scaling'):
    """
    Performs pre-processing of words for wordcloud:
    1. only headlines which are recommended (score > 0.5) are retained
    2. Only three most common words are retained
    3. Stopwords and short words are removed

    After that, either the words are counted, or they are scaled by the score of their headline and then summed
    :param scores: headline scores
    :param word_deviations: word scores
    :param mode: calculating or scaling
    :return:
    """
    c_word_deviations = Counter()
    word_deviations = [word_dict for word_dict, score in zip(word_deviations, scores) if score > 0.5]

    for i, headline_counter in enumerate(word_deviations):
        sorted_headline = Counter(headline_counter).most_common(3)
        sorted_headline = [(w, s) for (w, s) in sorted_headline if w not in STOPWORDS and len(w) >= 3]
        if mode == 'counting':
            words = [w for (w, s) in sorted_headline]
            c_word_deviations.update(words)
        elif mode == 'scaling':
            scaled_headline = [(w, s * score) for score, (w, s) in zip(scores, sorted_headline)]
            c_word_deviations += dict(scaled_headline)
    return generate_wordcloud(c_word_deviations)
