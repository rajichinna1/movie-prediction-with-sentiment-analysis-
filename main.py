import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request
from sklearn.neighbors import NearestNeighbors
import time
import bs4 as bs
import urllib.request
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.model_selection import train_test_split
from keras.layers import Input
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
# import the svm model and tfidf vectorizer
filename = 'svm_model.pkl'
svm = pickle.load(open(filename, 'rb'))
filenamee='recommend_function.pkl'
vectorizer = pickle.load(open('tranform.pkl','rb'))
new_df = pd.read_csv('mains_data.csv')

# converting list of string to list (eg. "["abc","def"]" to ["abc","def"])
def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["','')
    my_list[-1] = my_list[-1].replace('"]','')
    return my_list

# convert list of numbers to list (eg. "[1,2,3]" to [1,2,3])
def convert_to_list_num(my_list):
    my_list = my_list.split(',')
    my_list[0] = my_list[0].replace("[","")
    my_list[-1] = my_list[-1].replace("]","")
    return my_list
import urllib.request
import urllib.parse

def fetch_comments_and_sentiment(imdb_id):
    # Encode the movie title properly    
    # Set the custom user-agent
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    
    # Construct the URL
    url = f'https://www.imdb.com/title/{imdb_id}/reviews/?ref_=tt_ov_rt'
    try:
        # Send the request with custom headers
        request = urllib.request.Request(url, headers=headers)
        response = urllib.request.urlopen(request)
        html = response.read()

        soup = bs.BeautifulSoup(html, 'lxml')
        soup_result = soup.find_all("div", {"class": "text show-more__control"})
        
        reviews_list = []  # list of reviews
        reviews_status = []  # list of comments (good or bad)
        
        for reviews in soup_result:
            if reviews.string:
                reviews_list.append(reviews.string)
                # passing the review to our model
                movie_review_list = np.array([reviews.string])
                movie_vector = vectorizer.transform(movie_review_list)
                pred = svm.predict(movie_vector)
                reviews_status.append('Positive' if pred else 'Negative')

                time.sleep(1)
        # Combine reviews and sentiments into a dictionary
        movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}

        return movie_reviews
    except Exception as e:
        # Handle exceptions gracefully
        print(f"Error fetching comments and sentiment: {e}")
        return {}



def audience_prediciton(movie_name):
        df_ratings = pd.read_csv("movie_metadata.csv")


        original_title = df_ratings['movie_title']
        org_title = []
        #Removing white space
        for tile in original_title:
                org_title.append(tile.strip())

        df_ratings['movie_title'] = org_title


        ratings = df_ratings["imdb_score"]
        rating_list = []
        for rating in ratings:
                if rating >= 0 and rating <= 5.9:
                        rating_list.append(1)
                elif rating >= 6 and rating <= 6.9:
                        rating_list.append(2)
                elif rating >= 7 and rating <= 7.9:
                        rating_list.append(3)
                else:
                        rating_list.append(4)


        df_ratings['audience_class'] = rating_list


        df = df_ratings[['num_voted_users', 'imdb_score', 'audience_class']]


        x = df.drop('audience_class', axis=1)
        y = df['audience_class']

        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.15)


        from sklearn.ensemble import RandomForestClassifier
        mlp = MLPClassifier()
        mlp.fit(xtrain, ytrain)
        x_test=[]
        df_rv2 = df_ratings.loc[df_ratings['movie_title']==movie_name, ['num_voted_users', 'imdb_score']]
        for votes in df_rv2["num_voted_users"]:
                x_test.append(votes)
                break
        for score in df_rv2["imdb_score"]:
                
                x_test.append(score)
                break
        if(len(x_test)==0 ):
             return "N/A"
        predict_result = mlp.predict([x_test])

        if predict_result==1:
            result="Junior"
        elif predict_result==2:
            result="Teenage"
        elif predict_result == 3:
            result = "Mid-age"
        elif predict_result==4:
             result = "Senior"
        else:
           result = "N/A"
        return result

def movie_hit_prediciton(movie_name):
    df_ratings = pd.read_csv("movie_metadata.csv")

    original_title = df_ratings['movie_title']
    org_title = []
    # Removing white space
    for tile in original_title:
        org_title.append(tile.strip())

    df_ratings['movie_title'] = org_title

    ratings = df_ratings["imdb_score"]
    rating_list = []
    for rating in ratings:
        if rating >= 0 and rating <= 4.9:
            rating_list.append(0)
        elif rating >= 5 and rating <= 5.9:
            rating_list.append(1)
        elif rating >= 6 and rating <= 6.9:
            rating_list.append(2)
        elif rating >= 7 and rating <= 7.9:
            rating_list.append(3)
        elif rating >= 8 and rating <= 8.9:
            rating_list.append(4)
        else:
            rating_list.append(5)

    df_ratings['hit_class'] = rating_list

    x = df_ratings[['num_voted_users', 'imdb_score']]
    y = df_ratings['hit_class']

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.30)

    rf = RandomForestClassifier()
    rf.fit(xtrain, ytrain)

    df_rv2 = df_ratings.loc[df_ratings['movie_title']==movie_name, ['num_voted_users', 'imdb_score']]
    x_test = df_rv2.values
    if(len(x_test)==0):
         return 6
    
    pred_y = rf.predict(x_test)
    
    return pred_y[0]
def movie_popularity(movie_name):
    results=[]
    
    hpre=movie_hit_prediciton(movie_name)
    
    #hpre=hit_prediction[0]

    if hpre==0:
        result="F"
    elif hpre==1:
        result="A"
    elif hpre == 2:
        result = "AA"
    elif hpre == 3:
        result = "H"
    elif hpre == 4:
        result = "SH"
    elif hpre==5:
        result = "SDH"
    else:
         result="N/A"
    return result

def recommend_new(movie_content):
   
    x_test = [[movie_content]]

    # TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')

    
    tfidf_matrix = tfidf.fit_transform(new_df['tags'])

    # TF-IDF for the input movie overview
    tfidf_test = tfidf.transform(x_test[0])

    # Find nearest neighbors based on cosine similarity
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(tfidf_matrix)
    distances, indices = model_knn.kneighbors(tfidf_test, n_neighbors=6)  # Including the input movie itself

    # Extract movie information for recommended movies
    recommended_movies = []
    for i in range(1, len(distances.flatten())):  # Exclude the input movie
        index = indices.flatten()[i]
        recommended_movies.append(new_df.iloc[index]['movie_title'])  # Assuming 'original_title' is the correct column name
    
    return recommended_movies


app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html')


@app.route("/recommend",methods=["POST"])
def recommend():
   
    rec_movies = request.form['rec_movies']
    rec_posters = request.form['rec_posters']
    rec_movies_org = request.form['rec_movies_org']
    rec_year = request.form['rec_year']
    rec_vote = request.form['rec_vote']
    # for converting string to list
    rec_movies_org = convert_to_list(rec_movies_org)
    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
    rec_vote = convert_to_list_num(rec_vote)
    rec_year = convert_to_list_num(rec_year)
    popularity_movie=[]
    target_audienc=[]
    # rendering the string to python string
    for i in range(len(rec_movies)):
         print("movies:",rec_movies[i])
         popularity_movie.append(movie_popularity(rec_movies[i]))
    for i in range(len(rec_movies)):
         target_audienc.append(audience_prediciton(rec_movies[i]))
         
    # combining all list as a dictionary
    movie_cards = {rec_posters[i]: [rec_movies[i],rec_movies_org[i],rec_vote[i],rec_year[i],popularity_movie[i],target_audienc[i]] for i in range(len(rec_posters))}
  
    
    # passing all the data to the html file
    return render_template('recommend.html',movie_cards=movie_cards)

@app.route('/recommendations/<string:movie_id>')
def get_recommendations(movie_id):
    try:
    # Load recommendation data from pickle file
       
    # Get recommendations for the specified movie_id
       
    # Now you can use the recommends function without passing any arguments
        recommendations = recommend_new(movie_id)
       
    # Convert recommendations to JSON
        json_recommendations = jsonify(recommendations)
        return recommendations
    except Exception as e:
        # Return error response if there's an issue getting recommendations
        return jsonify({'error': str(e)})

@app.route('/sentiment/<imdb_id>', methods=['GET'])
def sentiment_analysis(imdb_id):
    # Access the movie title from query parameters
    movie_title = request.args.get('movie_title')
    # Perform sentiment analysis for the specified movie
    # Fetch comments and perform sentiment analysis
    # Prepare data for displaying
    #  comments and sentiment analysis
    
    # Assuming you have a function to fetch comments and perform sentiment analysis
    movie_reviews = fetch_comments_and_sentiment(imdb_id)
    
    # Pass comments and sentiment analysis to the template
    return render_template('sentiment_analysis.html',title=movie_title, reviews=movie_reviews)

if __name__ == '__main__':
    app.run(debug=True)