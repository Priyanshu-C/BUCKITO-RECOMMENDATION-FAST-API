# %%
import pandas as pd
import numpy as np

data= pd.read_csv("data\movies_metadata.csv")
data.head(5)



# %%

#Calculating the mean of vote avg
C = data['vote_average'].mean()


# Calculate the minimum number of votes to be considered
m = data['vote_count'].quantile(0.75)


# Filtering all the movies that have vote_count more than 'm'
quali = data.copy().loc[data['vote_count'] >= m]

# %%
#quali['Vote_avg'] = quali['Vote_avg'].astype(int)

# Calculating the weighted rating of each movie
def weighted_rate(v, R, m=m, C=C):
  ans = (v/(v+m) * R)+(m/(v+m) * C)
  return ans

quali['Score'] = quali.apply(lambda row : weighted_rate(row['vote_count'], row['vote_average']), axis = 1)

quali = quali.sort_values('Score', ascending=False)

conbas_df = quali[['original_title','genres','budget','overview']]

# %%

#converting the names and keyword instances into lowercase and strip all the spaces between them
#conbas_df['Title'] = conbas_df['Title'].str.replace(' ','').str.lower().str.replace('-','')
conbas_df['genres'] = conbas_df['genres'].str.replace(' ','').str.lower().str.replace('-','')

conbas_df['overview'] = conbas_df['overview'].replace(np.nan, 'Not Available')

conbas_df['soup'] = conbas_df['overview'] + ' ' + conbas_df['genres']


conbas_df = conbas_df.drop(columns=['budget'])

# %%
q = pd.merge(quali,conbas_df, left_on=['original_title'], right_on=['original_title'], how='left')

q = q.drop_duplicates(subset='imdb_id')


q[q['original_title'].str.contains('Scarface')]

q[q.duplicated(['original_title'])]

# q.to_csv('/content/Movies.csv')

# %%
# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(q['soup'])

count_matrix.shape

# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(count_matrix, count_matrix)

q = q.reset_index()
indices = pd.Series(q.index, index=q['original_title'])
# %%
# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim = cosine_sim):
    idx = indices[title]  # Get the index of the movie that matches the title
    sim_scores = list(enumerate(cosine_sim[idx]))   # Get the pairwsie similarity scores of all movies with that movie
    #print(sim_scores)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)  # Sort the movies based on the similarity scores
    #print(sim_scores)
    sim_scores = sim_scores[1:11]  # Get the scores of the 15 most similar movies
    #print(sim_scores)
    movie_indices = [i[0] for i in sim_scores]  # Get the movie indices

    movies = q.iloc[movie_indices][['original_title','Score']]  # Getting the weighted ratings of the movies 
    # Return the top 15 most similar movies arranged by ratings
    return movies.sort_values('Score', ascending = False)

get_recommendations('The Dark Knight Rises')
