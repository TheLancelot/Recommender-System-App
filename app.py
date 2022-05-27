import streamlit as st
import pickle
import requests
import pandas as pd
import numpy as np
#from main import genre_recommendations, title_recommendations

movies_dict=pickle.load(open('movie_dict.pkl','rb'))
movies=pd.DataFrame(movies_dict)
#similarity=pickle.load(open('similarity.pkl','rb'))
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

tf1 = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix1 = tf.fit_transform(movies['title'])

cosine_sim1 = cosine_similarity(tfidf_matrix1, tfidf_matrix1)
st.title("NKsirFLIX")

option = st.selectbox(
     'On what basis do you want to search?',
     ('Movies with Similar Genre','Title','Genre','Tag','What users similar to you liked','On Your History','User ID','All Time Top','Recent Top'))

#st.write('You selected:', option)
titles = movies['title']
indices = pd.Series(movies.index, index=movies['title'])

movies1=pd.read_csv('movies.csv')
movies1['released'] =movies1['title'].str.extract('.*\((.*)\).*',expand = False)
movies1['title']=movies1.title.str.slice(0,-7)

movie_score=pickle.load(open('movie_score.pkl','rb'))
movie_score=pd.DataFrame(movie_score)

link=pd.read_csv("links.csv")
link=link.dropna()
link['tmdbId'] = link['tmdbId'].astype(int)
#imilarity1=pickle.load(open('similarity1.pkl','rb'))

def genre_rec(title):#20 movies returned

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[0:21]
    movie_indices = [i[0] for i in sim_scores]
    #return titles.iloc[movie_indices]
    dfRec = pd.DataFrame(titles.iloc[movie_indices])
    # dfRec=dfRec.reset_index(drop=True)  
    # return dfRec.titlei
    df= pd.merge(dfRec,movies1,how='inner')
    df=pd.merge(df,link,how='inner')
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in range(10):
        # fetch the movie poster
        movie_id = df.iloc[i].tmdbId
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(df.iloc[i].title)

    return recommended_movie_names,recommended_movie_posters
def best_movies_by_genre(genre,top_n=10):
    df = pd.DataFrame(movie_score.loc[(movie_score[genre]==1)].sort_values(['weighted_score'],ascending=False)[['title','released','count','mean','weighted_score','tmdbId']])
    df = df.drop_duplicates(keep = "first")
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in range(10):
        # fetch the movie poster
        movie_id = df.iloc[i].tmdbId
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(df.iloc[i].title)

    return recommended_movie_names,recommended_movie_posters    
def title_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim1[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[0:21]
    movie_indices = [i[0] for i in sim_scores]
    #return titles.iloc[movie_indices]
    dfRec = pd.DataFrame(titles.iloc[movie_indices])
    # dfRec=dfRec.reset_index(drop=True)  
    # return dfRec.title
    df= pd.merge(dfRec,movies1,how='inner')
    df=pd.merge(df,link,how='inner')
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in range(10):
        # fetch the movie poster
        movie_id = df.iloc[i].tmdbId
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(df.iloc[i].title)

    return recommended_movie_names,recommended_movie_posters
def best_movies(tag):
#     d= pd.DataFrame(movie_score.loc[(movie_score["tag"]==tag)].sort_values(['weighted_score'],ascending=False)[['title','count','mean','weighted_score','tag']])
#     d = d.drop_duplicates(keep = "first")
    e = pd.DataFrame(movie_score.loc[(movie_score["tag"]==tag)].sort_values(['weighted_score'],ascending=False)[['title','tmdbId','count','mean','weighted_score','tag']])
    e = e.drop_duplicates(keep = "first")
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in range(10):
        # fetch the movie poster
        movie_id = e.iloc[i].tmdbId
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(e.iloc[i].title)

    return recommended_movie_names,recommended_movie_posters 

def top_movies(top_n=15):
    d = pd.DataFrame(movie_score.sort_values('weighted_score',ascending=False)[['title','tmdbId','count','mean','weighted_score']])
    d = d.drop_duplicates(keep = "first")
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in range(10):
        # fetch the movie poster
        movie_id = d.iloc[i].tmdbId
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(d.iloc[i].title)

    return recommended_movie_names,recommended_movie_posters 

def recent_top_movies(top_n=15):
    d = pd.DataFrame(movie_score.sort_values(by=['released','weighted_score'],ascending=False)[['title','tmdbId','released','count','mean','weighted_score']])
    d = d.drop_duplicates(keep = "first")
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in range(10):
        # fetch the movie poster
        movie_id = d.iloc[i].tmdbId
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(d.iloc[i].title)

    return recommended_movie_names,recommended_movie_posters

def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US".format(movie_id) 
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path

moviee=pickle.load(open('moviee.pkl','rb'))
moviee=pd.DataFrame(moviee)
rating_pivot=pickle.load(open('pivot.pkl','rb'))
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
        recommended_movie_names= [str(movies[movies['movieId']==mid]['title']).split('\n')[0].split('  ')[-1] for mid in movieids if mid not in [movieid]]
        rec=[]

        for i in range(len(recommended_movie_names)): 
            rec.append(list(movies[movies['title']==recommended_movie_names[i]]['tmdbId']))
        rec=[rec[0] for rec in rec]    
        recommended_movie_posters = []
        for i in rec:
        # fetch the movie poster
            recommended_movie_posters.append(fetch_poster(i))
        return recommended_movie_names,recommended_movie_posters
    
    # This method will recommend movies based on history stored in self.hist list
    def recommend_on_history(self,n_reccomend = 10):
        if self.ishist == False:
            return print('No history found')
        history = np.array([list(rating_pivot.loc[mid]) for mid in self.hist])
        distance,neighbors = nn_algo.kneighbors([np.average(history,axis=0)],n_neighbors=n_reccomend + len(self.hist))
        movieids = [rating_pivot.iloc[i].name for i in neighbors[0]]
        recommended_movie_names = [str(movies[movies['movieId']==mid]['title']).split('\n')[0].split('  ')[-1] for mid in movieids if mid not in self.hist]
        rec=[]

        for i in range(len(recommended_movie_names)): 
            rec.append(list(movies[movies['title']==recommended_movie_names[i]]['tmdbId']))
        rec=[rec[0] for rec in rec]    
        recommended_movie_posters = []
        for i in rec:
        # fetch the movie poster
            recommended_movie_posters.append(fetch_poster(i))
        return recommended_movie_names,recommended_movie_posters

recommender = Recommender()
moviees=pickle.load(open('moviees.pkl','rb'))
moviees=pd.DataFrame(moviees)

user_movie_df=pickle.load(open('user_movie_df.pkl','rb'))
user_movie_df=pd.DataFrame(user_movie_df)

df=pickle.load(open('df.pkl','rb'))
df=pd.DataFrame(df)

rating=pickle.load(open('rating.pkl','rb'))
rating=pd.DataFrame(rating)

import random
def movie_id_actual(random_user):
    movie_id_act = rating[(rating["userId"] == random_user) & (rating["rating"] >= 3.0)].\
        sort_values(by="timestamp", ascending=False)["movieId"][0:1].tolist()
    return movie_id_act[0]


def movie_id_name(dataframe, movie_ID):
    movie_name = dataframe[dataframe["movieId"] == movie_ID][["title"]].values[0].tolist()
    return movie_name[0]

def item_based_recommender(random_user):
    #movie_name, user_movie_df
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
    movie_to_recommend = random.sample(movie_to_recommend, 10)
    
    rec=[]

    for i in range(len(movie_to_recommend)): 
            rec.append(list(moviees[moviees['title']==movie_to_recommend[i]]['tmdbId']))  
    recommended_movie_posters = []
    rec=[rec[0] for rec in rec]  
    for i in rec:
        # fetch the movie poster
        recommended_movie_posters.append(fetch_poster(i))
    return movie_to_recommend,recommended_movie_posters

    
if (option=='Movies with Similar Genre'):
    title = st.text_input('Enter Movie Name')
    if st.button("Recommend"):
        recommended_movie_names,recommended_movie_posters = genre_rec(title) 
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.text(recommended_movie_names[0])
            st.image(recommended_movie_posters[0])
            st.text(recommended_movie_names[5])
            st.image(recommended_movie_posters[5]) 
        with col2:
                st.text(recommended_movie_names[1])
                st.image(recommended_movie_posters[1])
                st.text(recommended_movie_names[6])
                st.image(recommended_movie_posters[6]) 

        with col3:
                st.text(recommended_movie_names[2])
                st.image(recommended_movie_posters[2])
                st.text(recommended_movie_names[7])
                st.image(recommended_movie_posters[7])
        with col4:
                st.text(recommended_movie_names[3])
                st.image(recommended_movie_posters[3])
                st.text(recommended_movie_names[8])
                st.image(recommended_movie_posters[8])
        with col5:
                st.text(recommended_movie_names[4])
                st.image(recommended_movie_posters[4]) 
                st.text(recommended_movie_names[9])
                st.image(recommended_movie_posters[9])
elif(option=='Title'):
    title = st.text_input('Enter Movie Name')
    if st.button("Recommend"):
        recommended_movie_names,recommended_movie_posters = title_recommendations(title) 
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.text(recommended_movie_names[0])
            st.image(recommended_movie_posters[0])
            st.text(recommended_movie_names[5])
            st.image(recommended_movie_posters[5]) 
        with col2:
                st.text(recommended_movie_names[1])
                st.image(recommended_movie_posters[1])
                st.text(recommended_movie_names[6])
                st.image(recommended_movie_posters[6]) 

        with col3:
                st.text(recommended_movie_names[2])
                st.image(recommended_movie_posters[2])
                st.text(recommended_movie_names[7])
                st.image(recommended_movie_posters[7])
        with col4:
                st.text(recommended_movie_names[3])
                st.image(recommended_movie_posters[3])
                st.text(recommended_movie_names[8])
                st.image(recommended_movie_posters[8])
        with col5:
                st.text(recommended_movie_names[4])
                st.image(recommended_movie_posters[4]) 
                st.text(recommended_movie_names[9])
                st.image(recommended_movie_posters[9])
elif(option=='Genre'):
        title = st.text_input('Enter the Genre')
        if st.button("Recommend"):
            recommended_movie_names,recommended_movie_posters = best_movies_by_genre(title) 
            col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.text(recommended_movie_names[0])
            st.image(recommended_movie_posters[0])
            st.text(recommended_movie_names[5])
            st.image(recommended_movie_posters[5]) 
        with col2:
                st.text(recommended_movie_names[1])
                st.image(recommended_movie_posters[1])
                st.text(recommended_movie_names[6])
                st.image(recommended_movie_posters[6]) 

        with col3:
                st.text(recommended_movie_names[2])
                st.image(recommended_movie_posters[2])
                st.text(recommended_movie_names[7])
                st.image(recommended_movie_posters[7])
        with col4:
                st.text(recommended_movie_names[3])
                st.image(recommended_movie_posters[3])
                st.text(recommended_movie_names[8])
                st.image(recommended_movie_posters[8])
        with col5:
                st.text(recommended_movie_names[4])
                st.image(recommended_movie_posters[4]) 
                st.text(recommended_movie_names[9])
                st.image(recommended_movie_posters[9])
elif(option=='Tag'):
        tag = st.text_input('Enter tag')
        st.write('The current tag', tag)   
        if st.button("Recommend"):
            recommended_movie_names,recommended_movie_posters = best_movies(tag) 
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.text(recommended_movie_names[0])
                st.image(recommended_movie_posters[0])
                st.text(recommended_movie_names[5])
                st.image(recommended_movie_posters[5]) 
            with col2:
                    st.text(recommended_movie_names[1])
                    st.image(recommended_movie_posters[1])
                    st.text(recommended_movie_names[6])
                    st.image(recommended_movie_posters[6]) 

            with col3:
                    st.text(recommended_movie_names[2])
                    st.image(recommended_movie_posters[2])
                    st.text(recommended_movie_names[7])
                    st.image(recommended_movie_posters[7])
            with col4:
                    st.text(recommended_movie_names[3])
                    st.image(recommended_movie_posters[3])
                    st.text(recommended_movie_names[8])
                    st.image(recommended_movie_posters[8])
            with col5:
                    st.text(recommended_movie_names[4])
                    st.image(recommended_movie_posters[4]) 
                    st.text(recommended_movie_names[9])
                    st.image(recommended_movie_posters[9])   
elif(option=='What users similar to you liked'):
    title = st.text_input('Enter Movie Name')
    st.write('The current movie title is', title)  
    if st.button("Recommend"): 
        recommended_movie_names,recommended_movie_posters = recommender.recommend_on_movie(title)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.text(recommended_movie_names[0])
            st.image(recommended_movie_posters[0])
            st.text(recommended_movie_names[5])
            st.image(recommended_movie_posters[5]) 
        with col2:
                st.text(recommended_movie_names[1])
                st.image(recommended_movie_posters[1])
                st.text(recommended_movie_names[6])
                st.image(recommended_movie_posters[6]) 

        with col3:
                st.text(recommended_movie_names[2])
                st.image(recommended_movie_posters[2])
                st.text(recommended_movie_names[7])
                st.image(recommended_movie_posters[7])
        with col4:
                st.text(recommended_movie_names[3])
                st.image(recommended_movie_posters[3])
                st.text(recommended_movie_names[8])
                st.image(recommended_movie_posters[8])
        with col5:
                st.text(recommended_movie_names[4])
                st.image(recommended_movie_posters[4]) 
                st.text(recommended_movie_names[9])
                st.image(recommended_movie_posters[9]) 
elif(option=='On your history'):
    st.write('Movie Recommendation based on your history')  
    if st.button("Recommend"): 
        recommended_movie_names,recommended_movie_posters = recommender.recommend_on_history()
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.text(recommended_movie_names[0])
            st.image(recommended_movie_posters[0])
            st.text(recommended_movie_names[5])
            st.image(recommended_movie_posters[5]) 
        with col2:
                st.text(recommended_movie_names[1])
                st.image(recommended_movie_posters[1])
                st.text(recommended_movie_names[6])
                st.image(recommended_movie_posters[6]) 

        with col3:
                st.text(recommended_movie_names[2])
                st.image(recommended_movie_posters[2])
                st.text(recommended_movie_names[7])
                st.image(recommended_movie_posters[7])
        with col4:
                st.text(recommended_movie_names[3])
                st.image(recommended_movie_posters[3])
                st.text(recommended_movie_names[8])
                st.image(recommended_movie_posters[8])
        with col5:
                st.text(recommended_movie_names[4])
                st.image(recommended_movie_posters[4]) 
                st.text(recommended_movie_names[9])
                st.image(recommended_movie_posters[9])  
elif(option=='User ID'):
    title = st.text_input('Enter userid')
    st.write('The current userid is', title) 
    if st.button("Recommend"): 
        recommended_movie_names,recommended_movie_posters = item_based_recommender(title)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.text(recommended_movie_names[0])
            st.image(recommended_movie_posters[0])
            st.text(recommended_movie_names[5])
            st.image(recommended_movie_posters[5]) 
        with col2:
                st.text(recommended_movie_names[1])
                st.image(recommended_movie_posters[1])
                st.text(recommended_movie_names[6])
                st.image(recommended_movie_posters[6]) 

        with col3:
                st.text(recommended_movie_names[2])
                st.image(recommended_movie_posters[2])
                st.text(recommended_movie_names[7])
                st.image(recommended_movie_posters[7])
        with col4:
                st.text(recommended_movie_names[3])
                st.image(recommended_movie_posters[3])
                st.text(recommended_movie_names[8])
                st.image(recommended_movie_posters[8])
        with col5:
                st.text(recommended_movie_names[4])
                st.image(recommended_movie_posters[4]) 
                st.text(recommended_movie_names[9])
                st.image(recommended_movie_posters[9])   
elif(option=='All Time Top'):
    st.write('All time top movies')
    recommended_movie_names,recommended_movie_posters = top_movies()
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])
        st.text(recommended_movie_names[5])
        st.image(recommended_movie_posters[5]) 
    with col2:
            st.text(recommended_movie_names[1])
            st.image(recommended_movie_posters[1])
            st.text(recommended_movie_names[6])
            st.image(recommended_movie_posters[6]) 

    with col3:
            st.text(recommended_movie_names[2])
            st.image(recommended_movie_posters[2])
            st.text(recommended_movie_names[7])
            st.image(recommended_movie_posters[7])
    with col4:
            st.text(recommended_movie_names[3])
            st.image(recommended_movie_posters[3])
            st.text(recommended_movie_names[8])
            st.image(recommended_movie_posters[8])
    with col5:
            st.text(recommended_movie_names[4])
            st.image(recommended_movie_posters[4]) 
            st.text(recommended_movie_names[9])
            st.image(recommended_movie_posters[9])
elif(option=='Recent Top'):
    st.write('The top recently released movies are') 
    recommended_movie_names,recommended_movie_posters = recent_top_movies()
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])
        st.text(recommended_movie_names[5])
        st.image(recommended_movie_posters[5]) 
    with col2:
            st.text(recommended_movie_names[1])
            st.image(recommended_movie_posters[1])
            st.text(recommended_movie_names[6])
            st.image(recommended_movie_posters[6]) 

    with col3:
            st.text(recommended_movie_names[2])
            st.image(recommended_movie_posters[2])
            st.text(recommended_movie_names[7])
            st.image(recommended_movie_posters[7])
    with col4:
            st.text(recommended_movie_names[3])
            st.image(recommended_movie_posters[3])
            st.text(recommended_movie_names[8])
            st.image(recommended_movie_posters[8])
    with col5:
            st.text(recommended_movie_names[4])
            st.image(recommended_movie_posters[4]) 
            st.text(recommended_movie_names[9])
            st.image(recommended_movie_posters[9])  


