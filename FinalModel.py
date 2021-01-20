
# %%
from sklearn.feature_extraction.text import CountVectorizer
from tmdbv3api import TMDb, Movie
import requests
import json
import numpy as np
import pandas as pd
import re
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
import warnings
warnings.filterwarnings('ignore')

# %%
raw_data = pd.read_csv("data\movie_dataset.csv")

# %%
raw_data[raw_data.duplicated()]
df = raw_data[['genres', 'original_title', 'release_date', 'vote_average', 'vote_count', 'cast', 'director']]

# %%
# string date to Df fromate
df['release_date'] = pd.to_datetime(df['release_date'])
df['year'] = pd.DatetimeIndex(df['release_date']).year
df = df.drop('release_date', axis=1)

# %%
df = df.drop("year", axis=1)
df.isna().sum()

# %%
# Filing nan value with blank string
df['genres'] = df['genres'].replace(np.nan, " ")
df['cast'] = df['cast'].replace(np.nan, " ")
df['director'] = df['director'].replace(np.nan, " ")

# %%
pip install lxml

# %%
# Scraping 2018, 2019, 2020 mov data from wiki
link1 = "https://en.wikipedia.org/wiki/List_of_American_films_of_2018"
link2 = "https://en.wikipedia.org/wiki/List_of_American_films_of_2019"
link3 = "https://en.wikipedia.org/wiki/List_of_American_films_of_2020"

df1 = pd.read_html(link1, header=0)[2]
df2 = pd.read_html(link1, header=0)[3]
df3 = pd.read_html(link1, header=0)[4]
df4 = pd.read_html(link1, header=0)[5]

df5 = pd.read_html(link2, header=0)[3]
df6 = pd.read_html(link2, header=0)[4]
df7 = pd.read_html(link2, header=0)[5]
df8 = pd.read_html(link2, header=0)[6]

#######################################
df9 = pd.read_html(link3, header=0)[3]
df10 = pd.read_html(link3, header=0)[4]
df11 = pd.read_html(link3, header=0)[5]
df12 = pd.read_html(link3, header=0)[6]

# combine all scraped dataframe in one
frame = [df1, df2, df3, df4, df5, df6, df7, df8, df9, df10, df11, df12]
wiki_df = pd.concat(frame).reset_index(drop=True)
# %%
pip install tmdbv3api

# %%
#TMDB obj
tmdb = TMDb()
tmdb.api_key = '62fd9021dbeec142016bbfc8e3888baf'
tmdb_movie = Movie()
# This Function take movie title and return movie genre, vote_average, vote_count


def get_genre(title):
    try:
        result = tmdb_movie.search(title)
        movie_id = result[0].id
        response = requests.get(
            'https://api.themoviedb.org/3/movie/{}?api_key={}'.format(movie_id, tmdb.api_key))
        data_json = response.json()
        # Fetching
        movie_ids = []
        mids_str = " "
        for i in range(0, len(data_json['genres'])):
            movie_genres.append(data_json['genres'][i]['name'])
        genre = genr_str.join(movie_genres)

        # Fetching vote average and vote count
        vote_average = data_json['vote_average']
        vote_count = data_json['vote_count']
    except:
        vote_average = np.nan
        vote_count = np.nan
        genre = np.nan
    return genre, vote_average,vote_count


# %%
# columns...................
wiki_df['Genre'] = wiki_df['Title'].apply(lambda x: get_genre(x)[0])
wiki_df['Vote_Average'] = wiki_df['Title'].apply(lambda x: get_genre(x)[1])
wiki_df['Vote_Count'] = wiki_df['Title'].apply(lambda x: get_genre(x)[2])

# %%
wiki_df2 = wiki_df[["Title", "Cast and crew", "Genre", "Vote_Average", "Vote_Count"]]


# %%
wiki_df2[wiki_df2.isna().any(axis=1)]
wiki_df2 = wiki_df2.fillna({'Genre': '', 'Vote_Average': 0, 'Vote_Count': 0})
# %%
# Cheking random row values of cast and crew column.
sent_1 = wiki_df2['Cast and crew'][1]
sent_2 = wiki_df2['Cast and crew'][55]
sent_3 = wiki_df2['Cast and crew'][42]
sent_4 = wiki_df2['Cast and crew'][88]
sent_5 = wiki_df2['Cast and crew'][500]
print(sent_1)
print("====================================")
print(sent_2)
print("====================================")
print(sent_3)
print("====================================")
print(sent_4)
print("====================================")
print(sent_5)

# %%
preprocessed = []
for sentance in wiki_df2['Cast and crew'].values:

    # convert all uppercase-lowercase
    sentance = sentance.lower()
    sentance = re.sub(r"director", "", sentance)
    sentance = re.sub(r"co-director/screenplay", "", sentance)
    sentance = re.sub(r"co-/screenplay", "", sentance)
    sentance = re.sub(r"screenplay", "", sentance)
    sentance = re.sub(r"director/screenplay", "", sentance)

    sentance = re.sub('[^A-Za-z]+', ' ', sentance)
    sentance = re.sub(' +', ' ', sentance)
    # tokenizing the review by words
    sentance = sentance.split()
    sentance = ' '.join(sentance)
    # creating a corpus
    preprocessed.append(sentance)

wiki_df2["Cast and crew"] = preprocessed

# %%
#Now again check random row values of cast and crew column after cleaning.
sent_1 = wiki_df2['Cast and crew'][1]
sent_2 = wiki_df2['Cast and crew'][55]
sent_3 = wiki_df2['Cast and crew'][42]
sent_4 = wiki_df2['Cast and crew'][88]
sent_5 = wiki_df2['Cast and crew'][500]
sent_6 = wiki_df2['Cast and crew'][707]
print(sent_1)
print("====================================")
print(sent_2)
print("====================================")
print(sent_3)
print("====================================")
print(sent_4)
print("====================================")
print(sent_5)
print("====================================")
print(sent_6)
print("====================================")

# %%
# Text Preprocessing on movie genre column
preprocessed = []
for sentance in wiki_df2['Genre'].values:

    # convert all uppercase string into lowercase
    sentance = sentance.lower()

    # removing special symbol
    sentance = re.sub('[^A-Za-z]+', ' ', sentance)

    # removing extra space
    sentance = re.sub(' +', ' ', sentance)

    # tokenizing the review by words
    sentance = sentance.split()
    sentance = ' '.join(sentance)

    # creating a corpus
    preprocessed.append(sentance)  # Creating a corpus

wiki_df2["Genre"] = preprocessed

# %%
#Preprocessing on tittle..
preprocessed = []
for sentance in wiki_df2['Title'].values:
    sentance = sentance.lower()

    sentance = re.sub(' +', ' ', sentance)

    sentance = sentance.split()
    sentance = ' '.join(sentance)
    preprocessed.append(sentance)

wiki_df2["Title"] = preprocessed

# %%
df["Cast and crew"] = df['cast'] + ' ' + df['director']

df = df.rename({'original_title': 'Title', 'genres': 'Genre',
                'vote_average': 'Vote_Average', 'vote_count': 'Vote_Count'}, axis=1)
df['Title'] = df['Title'].str.lower()
df['Cast and crew'] = df['Cast and crew'].str.lower()
df['Genre'] = df['Genre'].str.lower()

df = df[['Title', 'Cast and crew', 'Genre', 'Vote_Average', 'Vote_Count']]

# %%
frame = [wiki_df2, df]
final_df = pd.concat(frame).reset_index(drop=True)
p_df = final_df[['Title', 'Vote_Average', 'Vote_Count']]

# %%
# Pura Model utha ke Copy paste :)
v = p_df['Vote_Count']
R = p_df['Vote_Average']
C = p_df['Vote_Average'].mean()
m = p_df['Vote_Count'].quantile(0.70)

p_df['Weighted_Average'] = ((R*v) + (C*m))/(v+m)
#weighted_average score
popular_movies = p_df.sort_values(by='Weighted_Average', ascending=False)

# %5
final_df['Combined_Features'] = final_df['Cast and crew'] + \
    ' ' + final_df['Genre']

# %%
# Create count matrix from this new combined column
cv = CountVectorizer()
count_matrix = cv.fit_transform(final_df["Combined_Features"])
# %%
#Cosine Similarity
cosine_sim = cosine_similarity(count_matrix)

# %%


def get_recomandation_contentBase(title):
    title = title.lower()
    title = get_close_matches(
        title, final_df['Title'].values, n=3, cutoff=0.6)[0]
    idx = final_df['Title'][final_df['Title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[0:16]

    for i in sim_scores:
        movie_index = i[0]
        print(final_df['Title'].iloc[movie_index])


# %%
get_recomandation_contentBase("tenet")

# %%
ratings = pd.read_csv("movies.csv")
movies = pd.read_csv("ratings.csv")
data = pd.merge(ratings, movies, on='movieId')
data = data[['movieId', 'title', 'userId', 'rating']]
n_users = data['userId'].nunique()
n_items = data['movieId'].nunique()
movie_rating_count = pd.DataFrame(data.groupby(
    'title')['rating'].count().reset_index())
movie_rating_count = movie_rating_count.rename(
    columns={'rating': 'total rating count'})
rating_with_totalRatingCount = pd.merge(data, movie_rating_count, on='title')
rating_popular_movie = rating_with_totalRatingCount[
    rating_with_totalRatingCount['total rating count'] >= 50]
movie_feature_df = rating_with_totalRatingCount.pivot_table(
    index='userId', columns='title', values='rating').fillna(0)
movie_feature_df = rating_with_totalRatingCount.pivot_table(index='userId',columns='title',values='rating').fillna(0)

# %%
user_similarity = movie_feature_df.corr()

# %5


def get_recomandation(movie_name, ratings):
    similar_score = user_similarity[movie_name]*(ratings-2.5)
    similar_score = similar_score.sort_values(ascending=False)

    return similar_score


# %%
item_similarity = cosine_similarity(movie_feature_df.T)
item_similarity_df = pd.DataFrame(item_similarity, index=movie_feature_df.columns,columns=movie_feature_df.columns)

# %%
# Function that takes in movie title and ratings as input and outputs most similar movies
def get_recomandation2(movie_name, ratings):
    similar_score = item_similarity_df[movie_name]*(ratings-2.5)
    similar_score = similar_score.sort_values(ascending=False)

    return similar_score
