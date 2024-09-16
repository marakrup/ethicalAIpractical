import configparser
import time
import streamlit as st
import umap
import sys

from src.clustering.AgglomerativeWrapper import AgglomorativeWrapper
from src.clustering.KMeansWrapper import KMeansWrapper
from src.recommendation.ClickPredictor import ClickPredictor, RankingModule
from src.utils import load_headlines, \
    generate_header, set_session_state, extract_unread, \
    get_wordcloud_from_attention, remove_old_files, reset_session_state

### GENERAL PAGE INFO ###

st.set_page_config(
    page_title="badpun",
    layout="wide"
)

generate_header()
remove_old_files()

if 'config' not in st.session_state:
    config = configparser.ConfigParser()
    config.read('config.ini')
    if len(sys.argv) > 1:
        if sys.argv[1] not in ['high', 'low']:
            raise ValueError(f"{sys.argv[1]} is not a valid command line parameter. Options are 'high' and 'low'")
        config['DEFAULT']['Dimensionality'] = sys.argv[1]
        print(f"Chosen dimensionality: {config['DEFAULT']['Dimensionality']}")
    st.session_state['config'] = config[config['DEFAULT']['Dimensionality']]

config = st.session_state['config']


### DATA LOADING ###
@st.cache_resource
def load_predictor():
    return ClickPredictor(huggingface_url="josh-oo/news-classifier",
                          commit_hash="1b0922bb88f293e7d16920e7ef583d05933935a9")


@st.cache_resource
def load_rm():
    return RankingModule(click_predictor)


@st.cache_resource
def fit_reducer():
    user_embedding = click_predictor.get_historic_user_embeddings()
    fit = umap.UMAP(
        n_neighbors=int(config['n_neighbors']),
        min_dist=float(config['min_dist']),
        n_components=int(config['n_components']),
        metric=config['metric']
    )
    return fit.fit(user_embedding)


@st.cache_resource
def get_model():
    """
    Creates and caches the model.
    :return:
    """
    embeddings = user_embedding
    model = KMeansWrapper(embeddings)
    return model


click_predictor = load_predictor()


@st.cache_data
def umap_transform():
    return reducer.transform(click_predictor.get_historic_user_embeddings())


ranking_module = load_rm()
reducer = fit_reducer()
user_embedding = umap_transform()
model = get_model()

set_session_state(user_embedding[112])

headlines = load_headlines()
unread_headlines_ind, unread_headlines = extract_unread(headlines)

prediction = model.predict(st.session_state.user)
# exemplars are the low dimensional medoids of the clusters
exemplars = user_embedding[model.repr_indeces]

##### TABS ####

cold_start_tab, recommendation_tab, alternative_tab = st.tabs(
    ["Reset User", "Personalized Recommendation", "Alternative Feeds"])

with cold_start_tab:
    st.write('To start off, choose a user which matches your interest most:')
    user_cols = st.columns(3)

    def choose_user(user_index, test):
        """
        Method resets the system to the cold start user, deletes old files and cleans session state
        :param user_index: the embedding index of the user in question
        :param test: is needed because the callback does not take a single argument
        """
        st.session_state['clean'] = False
        remove_old_files()
        reset_session_state(user_embedding[user_index])
        click_predictor.set_personal_user_embedding(user_index)


    for i, (col, user_index) in enumerate(zip(user_cols, [1228, 1700, 507])):
        # choice: 757/1228, 1700, 507;
        # food: 757,
        # celebrity: 1227,1228, 512;
        # politics: 751, 723, 517, 514, 510, 315, 1700, 495, 501, 502, 504, 750
        # sports: 1703, 507, 509, 720
        col.button(f"User {i + 1}", use_container_width=True, on_click=choose_user, args=(user_index, None),
                   type='primary')
        article_recommendations = ranking_module.rank_headlines(headlines.index, headlines.loc[:, 2], user_id=user_index,
                                                                take_top_k=10)

        article_fields = [col.button(f"[{headlines.loc[article_index, 1]}] {article}", use_container_width=True,
                                     key=f"{i}_{button_index}")
                          for button_index, (article, article_index, score) in
                          enumerate(article_recommendations)]

with recommendation_tab:
    ### LAYOUT ###
    left_column, right_column = st.columns([3, 1])
    news_tinder = left_column.container()

    lower_left, lower_right = st.columns(2)
    visualization = lower_left.container()
    interpretation = lower_right.container()

    ### 1. NEWS RECOMMENDATIONS ###
    article_recommendations = ranking_module.rank_headlines(unread_headlines_ind, unread_headlines, take_top_k=2)
    current_article = article_recommendations[0][0]
    current_index = article_recommendations[0][1]


    def handle_article(article_index, headline, read=1):
        """
        Mark article as handled an give feedback to model. Retrieves the new user embedding after the online learning
        step and updates the visualized embedding.
        :param article_index:
        :param headline:
        :param read: if the article is liked (1) or was skipped (0)
        """
        st.session_state.article_mask[article_index] = False
        click_predictor.update_step(headline, read)

        user = click_predictor.get_personal_user_embedding().reshape(1, -1)
        st.session_state.user = reducer.transform(user)[0]


    def read_later():
        """
        Simplified to just skipping the article but not passing any feedback to model
        """
        st.session_state.article_mask[current_index] = False


    news_tinder.subheader(f"[{headlines.loc[current_index, 1].capitalize()}] :blue[{current_article}]")

    ll, lm, lr = news_tinder.columns(3, gap='large')
    ll.button('Skip', use_container_width=True, on_click=handle_article, args=(current_index, current_article, 0))
    lm.button('Maybe later', use_container_width=True, on_click=read_later)
    lr.button('Read', use_container_width=True, on_click=handle_article, type="primary",
              args=(current_index, current_article, 1))

    ### 2. CLUSTERING ####
    visualization.header(f"You are assigned to cluster {prediction}")

    model.visualize(user_embedding, exemplars,
                    [("You", st.session_state.user), ("Initial profile", st.session_state.cold_start)])
    visualization.plotly_chart(model.figure, use_container_width=True)

    # ### 2.2. INTERPRETING ###
    interpretation.header('Interpretation')
    results = click_predictor.calculate_scores(list(headlines.loc[:, 2]))
    wordcloud = get_wordcloud_from_attention(*results)

    # Display the generated image:
    interpretation.image(wordcloud.to_array(), use_column_width="auto")

with alternative_tab:
    ### 1. CLUSTERING AND SUGGESTION ####
    left_column, right_column = st.columns(2)
    left_column.write(
        f"You have been matched with cluster {prediction}. Please feel free to choose any other cluster on the right.")
    left_column.write("Most clusters (such as cluster 3)"
                      f" are about murder, death, and "
                      f"calamities – oh well, human kind is just drawn to those big headlines. "
                      f"But there're also some clusters about sports, politics, celebrities, and food, as well as nicely "
                      f"mixed ones.")
    left_column.write(
        f" **We recommend to check out clusters 3, 5, 6, 8, 9, 13, and 14 to see some very clear cluster profiles**.")
    number = right_column.number_input('Cluster', min_value=0, max_value=int(config['NoClusters']) - 1,
                                       value=prediction)

    ### 2. PAGE LAYOUT ###
    left, middle, right = st.columns(3)

    ### 2.1 Newsfeed ###
    left.header('Newsfeed')


    def button_callback_alternative(article_index, test):
        "Mark article as read"
        st.session_state.article_mask[article_index] = False


    cluster_recommendations = ranking_module.rank_headlines(unread_headlines_ind, unread_headlines,
                                                            user_id=model.repr_indeces[number],
                                                            take_top_k=10)
    article_fields = [left.button(f"[{headlines.loc[article_index, 1]}] {article}", use_container_width=True,
                                  on_click=button_callback_alternative,
                                  args=(article_index, 0))
                      for button_index, (article, article_index, score) in
                      enumerate(cluster_recommendations)]  # sorry for ugly

    ### 2.2. Clustering ###

    middle.header('Clustering')
    model.visualize(user_embedding, exemplars,
                    [("Actual you", st.session_state.user),
                     ("Feed you are seeing", user_embedding[model.repr_indeces[number]])])
    middle.plotly_chart(model.figure)

    ### 2.3. INTERPRETATION ###
    right.header('Interpretation')

    explanation_method = right.radio(
        "Choose explanation method",
        ('LRP', 'Attention'), horizontal=True)

    # only load the precaluclated wordclouds if the config file is set to load AND the dimensionality is high
    # low dimensionality always caluclates live, as agglomorative clustering does not allow for deterministic clusters
    if config['WordcloudGeneration'] == 'load' and config['Dimensionality'] == 'high':
        right.image(f"media/{config['Dimensionality']}/{explanation_method.lower()}/scaling_{number}.svg",
                    use_column_width="auto")

    else:
        results = click_predictor.calculate_scores(list(headlines.loc[:, 2]), user_id=user_index)
        wordcloud = get_wordcloud_from_attention(*results)
        right.image(wordcloud.to_array(), use_column_width="auto")
