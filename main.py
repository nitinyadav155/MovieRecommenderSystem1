# import numpy as np
# import pandas as pd
# from flask import Flask, render_template, request
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import json
# import bs4 as bs
# import urllib.request
# import pickle
# import requests

# # load the nlp model and tfidf vectorizer from disk
# filename = 'nlp_model.pkl'
# clf = pickle.load(open(filename, 'rb'))
# vectorizer = pickle.load(open('tranform.pkl','rb'))

# def create_similarity():
#     data = pd.read_csv('main_data.csv')
#     # creating a count matrix
#     cv = CountVectorizer()
#     count_matrix = cv.fit_transform(data['comb'])
#     # creating a similarity score matrix
#     similarity = cosine_similarity(count_matrix)
#     return data,similarity

# def rcmd(m):
#     m = m.lower()
#     try:
#         data.head()
#         similarity.shape
#     except:
#         data, similarity = create_similarity()
#     if m not in data['movie_title'].unique():
#         return('Sorry! try another movie name')
#     else:
#         i = data.loc[data['movie_title']==m].index[0]
#         lst = list(enumerate(similarity[i]))
#         lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
#         lst = lst[1:11] # excluding first item since it is the requested movie itself
#         l = []
#         for i in range(len(lst)):
#             a = lst[i][0]
#             l.append(data['movie_title'][a])
#         return l
    
# # converting list of string to list (eg. "["abc","def"]" to ["abc","def"])

# def convert_to_list(my_list):
#     my_list = my_list.split('","')
#     my_list[0] = my_list[0].replace('["','')
#     my_list[-1] = my_list[-1].replace('"]','')
#     return my_list


# # to get suggestions of movies
# def get_suggestions():
#     data = pd.read_csv('main_data.csv')
#     return list(data['movie_title'].str.capitalize())

# # Flask API

# app = Flask(__name__)

# @app.route("/")
# @app.route("/home")
# def home():
#     suggestions = get_suggestions()
#     return render_template('home.html',suggestions=suggestions)

# @app.route("/similarity",methods=["POST"])
# def similarity():
#     movie = request.form['name']
#     rc = rcmd(movie)
#     if type(rc)==type('string'):
#         return rc
#     else:
#         m_str="---".join(rc)
#         return m_str
    

#     # Fetch movie details, including the trailer link
   

# @app.route("/recommend",methods=["POST"])
# def recommend():
#     # getting data from AJAX request
#     title = request.form['title']
#     cast_ids = request.form['cast_ids']
#     cast_names = request.form['cast_names']
#     cast_chars = request.form['cast_chars']
#     cast_bdays = request.form['cast_bdays']
#     cast_bios = request.form['cast_bios']
#     cast_places = request.form['cast_places']
#     cast_profiles = request.form['cast_profiles']
#     imdb_id = request.form['imdb_id']
#     poster = request.form['poster']
#     genres = request.form['genres']
#     overview = request.form['overview']
#     vote_average = request.form['rating']
#     vote_count = request.form['vote_count']
#     release_date = request.form['release_date']
#     runtime = request.form['runtime']
#     status = request.form['status']
#     rec_movies = request.form['rec_movies']
#     rec_posters = request.form['rec_posters']
#     movie_trailer = request.get('trailer_url')

#     # get movie suggestions for auto complete
#     suggestions = get_suggestions()

#     # call the convert_to_list function for every string that needs to be converted to list
#     rec_movies = convert_to_list(rec_movies)
#     rec_posters = convert_to_list(rec_posters)
#     cast_names = convert_to_list(cast_names)
#     cast_chars = convert_to_list(cast_chars)
#     cast_profiles = convert_to_list(cast_profiles)
#     cast_bdays = convert_to_list(cast_bdays)
#     cast_bios = convert_to_list(cast_bios)
#     cast_places = convert_to_list(cast_places)
    
#     # convert string to list (eg. "[1,2,3]" to [1,2,3])
#     cast_ids = cast_ids.split(',')
#     cast_ids[0] = cast_ids[0].replace("[","")
#     cast_ids[-1] = cast_ids[-1].replace("]","")
    
#     # rendering the string to python string
#     for i in range(len(cast_bios)):
#         cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"','\"')
    
#     # combining multiple lists as a dictionary which can be passed to the html file so that it can be processed easily and the order of information will be preserved
#     movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}

#     casts = {cast_names[i]:[cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}

#     cast_details = {cast_names[i]:[cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in range(len(cast_places))}

#     # web scraping to get user reviews from IMDB site
#     sauce = urllib.request.urlopen('https://www.imdb.com/title/{}/reviews?ref_=tt_ov_rt'.format(imdb_id)).read()
#     soup = bs.BeautifulSoup(sauce,'lxml')
#     soup_result = soup.find_all("div",{"class":"text show-more__control"})

#     reviews_list = [] # list of reviews
#     reviews_status = [] # list of comments (good or bad)
#     for reviews in soup_result:
#         if reviews.string:
#             reviews_list.append(reviews.string)
#             # passing the review to our model
#             movie_review_list = np.array([reviews.string])
#             movie_vector = vectorizer.transform(movie_review_list)
#             pred = clf.predict(movie_vector)
#             reviews_status.append('Good' if pred else 'Bad')

#     # combining reviews and comments into a dictionary
#     movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}     

#     # passing all the data to the html file
#     return render_template('recommend.html',title=title,poster=poster,overview=overview,vote_average=vote_average,
#         vote_count=vote_count,release_date=release_date,runtime=runtime,status=status,genres=genres,
#         movie_cards=movie_cards,reviews=movie_reviews,casts=casts,cast_details=cast_details,movie_trailer=movie_trailer)


# if __name__ == '__main__':
#     app.run(debug=True)


# import numpy as np
# import pandas as pd
# from flask import Flask, render_template, request
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import json
# import bs4 as bs
# import urllib.request
# import pickle
# import requests

# # Load the NLP model and TF-IDF vectorizer from disk
# filename = 'nlp_model.pkl'
# clf = pickle.load(open(filename, 'rb'))
# vectorizer = pickle.load(open('tranform.pkl', 'rb'))

# def create_similarity():
#     data = pd.read_csv('main_data.csv')
#     # Creating a count matrix
#     cv = CountVectorizer()
#     count_matrix = cv.fit_transform(data['comb'])
#     # Creating a similarity score matrix
#     similarity = cosine_similarity(count_matrix)
#     return data, similarity

# def rcmd(m):
#     m = m.lower()
#     try:
#         data.head()
#         similarity.shape
#     except:
#         data, similarity = create_similarity()
#     if m not in data['movie_title'].unique():
#         return('Sorry! try another movie name')
#     else:
#         i = data.loc[data['movie_title'] == m].index[0]
#         lst = list(enumerate(similarity[i]))
#         lst = sorted(lst, key=lambda x: x[1], reverse=True)
#         lst = lst[1:11]  # Excluding first item since it is the requested movie itself
#         l = []
#         for i in range(len(lst)):
#             a = lst[i][0]
#             l.append(data['movie_title'][a])
#         return l
    
# def convert_to_list(my_list):
#     my_list = my_list.split('","')
#     my_list[0] = my_list[0].replace('["', '')
#     my_list[-1] = my_list[-1].replace('"]', '')
#     return my_list

# def get_suggestions():
#     data = pd.read_csv('main_data.csv')
#     return list(data['movie_title'].str.capitalize())

# app = Flask(__name__)

# @app.route("/")
# @app.route("/home")
# def home():
#     suggestions = get_suggestions()
#     return render_template('home.html', suggestions=suggestions)

# @app.route("/similarity", methods=["POST"])
# def similarity():
#     movie = request.form['name']
#     rc = rcmd(movie)
#     if type(rc) == type('string'):
#         return rc
#     else:
#         m_str = "---".join(rc)
#         return m_str

# @app.route("/recommend", methods=["POST"])
# def recommend():
#     # Getting data from AJAX request
#     title = request.form['title']
#     cast_ids = request.form['cast_ids']
#     cast_names = request.form['cast_names']
#     cast_chars = request.form['cast_chars']
#     cast_bdays = request.form['cast_bdays']
#     cast_bios = request.form['cast_bios']
#     cast_places = request.form['cast_places']
#     cast_profiles = request.form['cast_profiles']
#     imdb_id = request.form['imdb_id']
#     poster = request.form['poster']
#     genres = request.form['genres']
#     overview = request.form['overview']
#     vote_average = request.form['rating']
#     vote_count = request.form['vote_count']
#     release_date = request.form['release_date']
#     runtime = request.form['runtime']
#     status = request.form['status']
#     rec_movies = request.form['rec_movies']
#     rec_posters = request.form['rec_posters']
#     # movie_trailer = request.form.get('trailer_url', '')

#     # Get movie suggestions for auto-complete
#     suggestions = get_suggestions()

#     # Call the convert_to_list function for every string that needs to be converted to a list
#     rec_movies = convert_to_list(rec_movies)
#     rec_posters = convert_to_list(rec_posters)
#     cast_names = convert_to_list(cast_names)
#     cast_chars = convert_to_list(cast_chars)
#     cast_profiles = convert_to_list(cast_profiles)
#     cast_bdays = convert_to_list(cast_bdays)
#     cast_bios = convert_to_list(cast_bios)
#     cast_places = convert_to_list(cast_places)
    
#     # Convert string to list (e.g., "[1,2,3]" to [1,2,3])
#     cast_ids = cast_ids.split(',')
#     cast_ids[0] = cast_ids[0].replace("[", "")
#     cast_ids[-1] = cast_ids[-1].replace("]", "")
    
#     # Rendering the string to Python string
#     for i in range(len(cast_bios)):
#         cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"', '\"')
    
#     # Combining multiple lists as a dictionary which can be passed to the HTML file so that it can be processed easily and the order of information will be preserved
#     movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}
#     casts = {cast_names[i]: [cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}
#     cast_details = {cast_names[i]: [cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in range(len(cast_places))}

#     # Web scraping to get user reviews from IMDB site
#     sauce = urllib.request.urlopen('https://www.imdb.com/title/{}/reviews?ref_=tt_ov_rt'.format(imdb_id)).read()
#     soup = bs.BeautifulSoup(sauce, 'lxml')
#     soup_result = soup.find_all("div", {"class": "text show-more__control"})

#     reviews_list = []  # List of reviews
#     reviews_status = []  # List of comments (good or bad)
#     for reviews in soup_result:
#         if reviews.string:
#             reviews_list.append(reviews.string)
#             # Passing the review to our model
#             movie_review_list = np.array([reviews.string])
#             movie_vector = vectorizer.transform(movie_review_list)
#             pred = clf.predict(movie_vector)
#             reviews_status.append('Good' if pred else 'Bad')

#     # Combining reviews and comments into a dictionary
#     movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}     

#     # Passing all the data to the HTML file
#     return render_template('recommend.html', title=title, poster=poster, overview=overview, vote_average=vote_average,
#         vote_count=vote_count, release_date=release_date, runtime=runtime, status=status, genres=genres,
#         movie_cards=movie_cards, reviews=movie_reviews, casts=casts, cast_details=cast_details)

# if __name__ == '__main__':
    # app.run(debug=True)
# 





import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import bs4 as bs
import urllib.request
import pickle
import requests

# Load the NLP model and TF-IDF vectorizer from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('tranform.pkl', 'rb'))

TMDB_API_KEY = 'YOUR_TMDB_API_KEY'

def create_similarity():
    data = pd.read_csv('main_data.csv')
    # Creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # Creating a similarity score matrix
    similarity = cosine_similarity(count_matrix)
    return data, similarity

def rcmd(m):
    m = m.lower()
    try:
        data.head()
        similarity.shape
    except:
        data, similarity = create_similarity()
    if m not in data['movie_title'].unique():
        return('Sorry! try another movie name')
    else:
        i = data.loc[data['movie_title'] == m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key=lambda x: x[1], reverse=True)
        lst = lst[1:11]  # Excluding first item since it is the requested movie itself
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['movie_title'][a])
        return l
    
def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["', '')
    my_list[-1] = my_list[-1].replace('"]', '')
    return my_list

def get_suggestions():
    data = pd.read_csv('main_data.csv')
    return list(data['movie_title'].str.capitalize())

def get_movie_trailer_id(api_key, movie_name):
    # Step 1: Search for the movie
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_name}"
    search_response = requests.get(search_url).json()
    if not search_response['results']:
        return None
    movie_id = search_response['results'][0]['id']

    # Step 2: Get the trailer details
    trailer_url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={api_key}"
    trailer_response = requests.get(trailer_url).json()
    trailers = [video for video in trailer_response['results'] if video['type'] == 'Trailer' and video['site'] == 'YouTube']
    
    if not trailers:
        return None
    
    # Return the YouTube video ID of the first trailer
    return trailers[0]['key']

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html', suggestions=suggestions)

@app.route("/similarity", methods=["POST"])
def similarity():
    movie = request.form['name']
    rc = rcmd(movie)
    if type(rc) == type('string'):
        return rc
    else:
        m_str = "---".join(rc)
        return m_str

@app.route("/recommend", methods=["POST"])
def recommend():
    # Getting data from AJAX request
    title = request.form['title']
    cast_ids = request.form['cast_ids']
    cast_names = request.form['cast_names']
    cast_chars = request.form['cast_chars']
    cast_bdays = request.form['cast_bdays']
    cast_bios = request.form['cast_bios']
    cast_places = request.form['cast_places']
    cast_profiles = request.form['cast_profiles']
    imdb_id = request.form['imdb_id']
    poster = request.form['poster']
    genres = request.form['genres']
    overview = request.form['overview']
    vote_average = request.form['rating']
    vote_count = request.form['vote_count']
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    status = request.form['status']
    rec_movies = request.form['rec_movies']
    rec_posters = request.form['rec_posters']

    # Get movie suggestions for auto-complete
    suggestions = get_suggestions()

    # Call the convert_to_list function for every string that needs to be converted to a list
    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
    cast_names = convert_to_list(cast_names)
    cast_chars = convert_to_list(cast_chars)
    cast_profiles = convert_to_list(cast_profiles)
    cast_bdays = convert_to_list(cast_bdays)
    cast_bios = convert_to_list(cast_bios)
    cast_places = convert_to_list(cast_places)
    
    # Convert string to list (e.g., "[1,2,3]" to [1,2,3])
    cast_ids = cast_ids.split(',')
    cast_ids[0] = cast_ids[0].replace("[", "")
    cast_ids[-1] = cast_ids[-1].replace("]", "")
    
    # Rendering the string to Python string
    for i in range(len(cast_bios)):
        cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"', '\"')
    
    # Combining multiple lists as a dictionary which can be passed to the HTML file so that it can be processed easily and the order of information will be preserved
    movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}
    casts = {cast_names[i]: [cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}
    cast_details = {cast_names[i]: [cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in range(len(cast_places))}

    # Web scraping to get user reviews from IMDB site
    sauce = urllib.request.urlopen('https://www.imdb.com/title/{}/reviews?ref_=tt_ov_rt'.format(imdb_id)).read()
    soup = bs.BeautifulSoup(sauce, 'lxml')
    soup_result = soup.find_all("div", {"class": "text show-more__control"})

    reviews_list = []  # List of reviews
    reviews_status = []  # List of comments (good or bad)
    for reviews in soup_result:
        if reviews.string:
            reviews_list.append(reviews.string)
            # Passing the review to our model
            movie_review_list = np.array([reviews.string])
            movie_vector = vectorizer.transform(movie_review_list)
            pred = clf.predict(movie_vector)
            reviews_status.append('Good' if pred else 'Bad')

    # Combining reviews and comments into a dictionary
    movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}     

    # Fetch the trailer ID
    trailer_id = get_movie_trailer_id('86ab661de020b979d1047ce7160d3526', title)

    # Passing all the data to the HTML file
    return render_template('recommend.html', title=title, poster=poster, overview=overview, vote_average=vote_average,
        vote_count=vote_count, release_date=release_date, runtime=runtime, status=status, genres=genres,
        movie_cards=movie_cards, reviews=movie_reviews, casts=casts, cast_details=cast_details, trailer_id=trailer_id)

if __name__ == '__main__':
    app.run(debug=True)
