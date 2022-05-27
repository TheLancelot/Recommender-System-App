from typing import Union
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import random


# Reading movies file
movies = pd.read_csv('movies.csv', sep=',', encoding='latin-1', usecols=['title', 'genres'])

movies['released'] =movies['title'].str.extract('.*\((.*)\).*',expand = False)
movies['title']=movies.title.str.slice(0,-7)

# Break up the big genre string into a string array
movies['genres'] = movies['genres'].str.split('|')
# Convert genres to string value
movies['genres'] = movies['genres'].fillna("").astype('str')

#Recomendation movies on inputted movie genres
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(movies['genres'])
tfidf_matrix.shape

from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Build a 1-dimensional array with movie titles
titles = movies['title']
indices = pd.Series(movies.index, index=movies['title'])

# Function that get movie recommendations based on the cosine similarity score of movie genres
def genre_recommendations(title):#20 movies returned

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[0:21]
    movie_indices = [i[0] for i in sim_scores]
    #return titles.iloc[movie_indices]
    dfRec = pd.DataFrame(titles.iloc[movie_indices])
    dfRec=dfRec.reset_index(drop=True)  
    return dfRec.title

tf1 = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix1 = tf.fit_transform(movies['title'])

cosine_sim1 = cosine_similarity(tfidf_matrix1, tfidf_matrix1)

# Build a 1-dimensional array with movie titles
titles1 = movies['title']
indices1 = pd.Series(movies.index, index=movies['title'])

# Function that get movie recommendations based on the cosine similarity score of movie genres
def title_recommendations(title):
    idx = indices1[title]
    sim_scores = list(enumerate(cosine_sim1[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[0:21]
    movie_indices = [i[0] for i in sim_scores]
    #return titles.iloc[movie_indices]
    dfRec = pd.DataFrame(titles.iloc[movie_indices])
    dfRec=dfRec.reset_index(drop=True)  
    return dfRec.title
''' -----------------------------------------------------------------------------------------------------------------'''
ratings=pd.read_csv("ratings.csv")
ratings_df = ratings.groupby(['userId','movieId']).aggregate(np.max)
movie_list=pd.read_csv("movies.csv")
tags=pd.read_csv('tags.csv')
genres = movie_list['genres']
movie_list['released'] =movie_list['title'].str.extract('.*\((.*)\).*',expand = False)
movie_list['title']=movie_list.title.str.slice(0,-7)
genre_list = ""
for index,row in movie_list.iterrows():
        genre_list += row.genres + "|"
#split the string into a list of values
genre_list_split = genre_list.split('|')
#de-duplicate values
new_list = list(set(genre_list_split))
#remove the value that is blank
new_list.remove('')
for genre in new_list :
    movie_list[genre] = movie_list.apply(lambda _:int(genre in _.genres), axis = 1)
no_of_users = len(ratings['userId'].unique())
no_of_movies = len(ratings['movieId'].unique())
sparsity = round(1.0 - len(ratings)/(1.0*(no_of_movies*no_of_users)),3)
avg_movie_rating = pd.DataFrame(ratings.groupby('movieId')['rating'].agg(['mean','count']))
avg_movie_rating['movieId']= avg_movie_rating.index
#Get the average movie rating across all movies 
avg_rating_all=ratings['rating'].mean()
avg_rating_all
#set a minimum threshold for number of reviews that the movie has to have
min_reviews=10
movie_score = avg_movie_rating.loc[avg_movie_rating['count']>min_reviews]
#Creating a function for weighted rating score based on the count of reviews
def weighted_rating(x, m=min_reviews, C=avg_rating_all):
    v = x['count']
    R = x['mean']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)
movie_score['weighted_score'] = movie_score.apply(weighted_rating, axis=1)    
movie_score = movie_score.drop(columns = 'movieId')
movie_score.reset_index()
movie_score = pd.merge(movie_score,movie_list,on='movieId')
movie_score = pd.merge(movie_score,tags,on='movieId')

def top_movies(top_n=15):
    d = pd.DataFrame(movie_score.sort_values('weighted_score',ascending=False)[['title','count','mean','weighted_score']])
    d = d.drop_duplicates(keep = "first")
    return d[:top_n].title

def recent_top_movies(top_n=15):
    d = pd.DataFrame(movie_score.sort_values(by=['released','weighted_score'],ascending=False)[['title','released','count','mean','weighted_score']])
    d = d.drop_duplicates(keep = "first")
    return d[:top_n].title

def best_movies_by_genre(genre,top_n=10):
    d = pd.DataFrame(movie_score.loc[(movie_score[genre]==1)].sort_values(['weighted_score'],ascending=False)[['title','released','count','mean','weighted_score']])
    d = d.drop_duplicates(keep = "first")
    return d[:top_n].title

def best_movies(tag):
    d= pd.DataFrame(movie_score.loc[(movie_score["tag"]==tag)].sort_values(['weighted_score'],ascending=False)[['title','count','mean','weighted_score','tag']])
    d = d.drop_duplicates(keep = "first")
    e = pd.DataFrame(movie_score.loc[(movie_score["tag"]==tag)].sort_values(['weighted_score'],ascending=False)[['title','count','mean','weighted_score','tag']])
    e = e.drop_duplicates(keep = "first")
    e.append(d)
    return e.title
''' -------------------------------------------------------------------------------------------------------------------------------------'''
#User Based CF

movies=pd.read_csv("movies.csv")
tags=pd.read_csv('tags.csv')
ratings=pd.read_csv("ratings.csv")
links=pd.read_csv('links.csv')
movies['released'] =movies['title'].str.extract('.*\((.*)\).*',expand = False)
movies['title']=movies.title.str.slice(0,-7)

rating_pivot = ratings.pivot_table(values='rating',columns='userId',index='movieId').fillna(0)
from sklearn.neighbors import NearestNeighbors
nn_algo = NearestNeighbors(metric='cosine')
nn_algo.fit(rating_pivot)

class Recommender:
    def __init__(self):
        # This list will stored movies that called atleast ones using recommend_on_movie method
        self.hist = [] 
        self.ishist = False # Check if history is empty
    
    # This method will recommend movies based on a movie that passed as the parameter
    def recommend_on_movie(self,movie,n_reccomend = 10):
        self.ishist = True
        movieid = int(movies[movies['title']==movie]['movieId'])
        self.hist.append(movieid)
        distance,neighbors = nn_algo.kneighbors([rating_pivot.loc[movieid]],n_neighbors=n_reccomend+1)
        movieids = [rating_pivot.iloc[i].name for i in neighbors[0]]
        recommeds = [str(movies[movies['movieId']==mid]['title']).split('\n')[0].split('  ')[-1] for mid in movieids if mid not in [movieid]]
        return recommeds[:n_reccomend]
    
    # This method will recommend movies based on history stored in self.hist list
    def recommend_on_history(self,n_reccomend = 10):
        if self.ishist == False:
            print('No history found')
            return 1; 
        history = np.array([list(rating_pivot.loc[mid]) for mid in self.hist])
        distance,neighbors = nn_algo.kneighbors([np.average(history,axis=0)],n_neighbors=n_reccomend + len(self.hist))
        movieids = [rating_pivot.iloc[i].name for i in neighbors[0]]
        recommeds = [str(movies[movies['movieId']==mid]['title']).split('\n')[0].split('  ')[-1] for mid in movieids if mid not in self.hist]
        return recommeds[:n_reccomend]

recommender = Recommender()  

''' -------------------------------------------------------------------------------------------------------------------------------------'''
#Item Based CF
movie = pd.read_csv("movies.csv")
rating = pd.read_csv("ratings.csv")
df_ = movie.merge(rating, how="left", on="movieId")
df = df_.copy()

def create_user_movie_df():
    rating_counts = pd.DataFrame(df["title"].value_counts())
    rating_counts.describe([0.05, 0.50, 0.75, 0.85, 0.90, 0.95, 0.99]).T
    limit = rating_counts.quantile([0.90]).T
    limit_90 = limit[0.9][0]
    rare_movies = rating_counts[rating_counts["title"] <= limit_90].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index="userId", columns="title", values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()

def movie_id_actual(random_user):
    movie_id_act = rating[(rating["userId"] == random_user) & (rating["rating"] >= 4.0)]. \
        sort_values(by="timestamp", ascending=False)["movieId"][0:1].tolist()
    return movie_id_act[0]

def movie_id_name(dataframe, movie_ID):
    movie_name = dataframe[dataframe["movieId"] == movie_ID][["title"]].values[0].tolist()
    return movie_name[0]

def item_based_recommender(random_user):
    movie_id=movie_id_actual(random_user)
    movie_name = movie_id_name(df, movie_id)
    movie_name = user_movie_df[movie_name]
    recommend_list = user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(20)
    recommend_list = recommend_list.reset_index()
    recommend_list = recommend_list["title"].tolist()
    recommend_list

    random_user_df = user_movie_df[user_movie_df.index == random_user]
    random_user_movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
    random_user_movies_watched_df = user_movie_df[random_user_movies_watched]
    random_user_movies_watched_df = random_user_movies_watched_df.columns.tolist()

    movie_to_recommend = [i for i in random_user_movies_watched_df if i not in recommend_list]
    movie_to_recommend = random.sample(movie_to_recommend, 5)
    return movie_to_recommend
''' -------------------------------------------------------------------------------------------------------------------------------------'''
app=FastAPI()

@app.get("/movie-genre-related/{mov}")
def movies_with_similar_genre_to_input_movie(mov):
     #mov=mov.title()
     mov=mov.strip()
     return list(genre_recommendations(mov))

@app.get("/title-based/{mov}")
def movies_with_similar_title_to_input_movie(mov):
    #mov=mov.title()
    #mov=mov.strip()
    return title_recommendations(mov).head(10)

@app.get("/genre-based/{genre}")
def movie_recommendation_on_input_genre(genre):
    if(genre=='IMAX'):
        genre=genre.strip()
        return best_movies_by_genre(genre)

    genre=genre.title()
    genre=genre.strip()

    return best_movies_by_genre(genre)

@app.get("/tag-based/{tag}")
def movie_recommendation_on_input_tag(tag):
    tag=tag.strip()
    return best_movies(tag)

@app.get('/user-based/{mov}')
def on_movie(mov):
    #mov=mov.title()
    mov=mov.strip()
    return recommender.recommend_on_movie(mov)

@app.post('/user-based')
def on_history():
    if(recommender.recommend_on_history()==1):
        return {"message": "No Movie History"}

    else:
        return recommender.recommend_on_history()

@app.get('/item-based/{userid}')
def on_userid(userid):
    return item_based_recommender(userid)   

@app.post('/top-movies')
def all_time_top_movies():
    return list(top_movies())

@app.post('/recent-top-movies')
def latest_top_movies():
    return list(recent_top_movies())