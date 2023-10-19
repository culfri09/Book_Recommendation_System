from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

'''
---Recommendation Algorithm---
'''

# Loads data set into data frame.
df = pd.read_csv('books.csv') 


#Data preprocessing: creates a huge matrix in which every word is represented by a number (the weight depends in the frequency). Stop words are not included.
tfidf_vectorizer = TfidfVectorizer(stop_words='english') #Initializes the TfidfVectorizer with stop words removed.
tfidf_matrix = tfidf_vectorizer.fit_transform(df['description'].fillna('')) # Fits and transform the data set from the .csv (just the description) to create the matrix.

# Computes cosine similarities
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix) # List which contains the distance between ALL the book's descriptions within the matrix. 

# Method to obtain the book recommendations. Takes as parameters the book's name and the number of recommendations that will be given. 
def get_recommendations(book_title, num_recommendations=5):
    book_index = df.index[df['title'] == book_title].tolist() # Finds the index of the book within the list. 
    if not book_index:
        return [] #If title is not within the df, it returns an empty recommendation.
    
    book_index = book_index[0]
    
    # Obtains the similitude scores between the input book with the rest of the books in the df. 
    sim_scores = list(enumerate(cosine_similarities[book_index])) #Creates a list with all of the distances between the input book and the ones in the df.

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) # Sorts books based on similitude score (from highest to lowest).
    
    recommendations = sim_scores[1:num_recommendations+1] #Creates list with the first 5 items of the sim_scores (list with the sorted books by similitude)
    
    #recommended_books = [(df['title'][rec[0]], rec[1]) for rec in recommendations]
    recommended_books = [(df['title'][rec[0]], df['description'][rec[0]], df['author'][rec[0]], df['publishDate'][rec[0]], rec[1]) for rec in recommendations] #Transforms the numbers of the recommendations into strings and adds them to recommendation_books list: book title + description + author + publish date
    return recommended_books 

'''
---Flask integration---
'''

app = Flask(__name__, template_folder='./template') #Creates a Flask application named app, and specifies path where the rendered HTML pages are located.

#Loads main page when user accesses the root of the URL.
@app.route('/')
def index():
    return render_template('index.html')

#Loads recommendation page when user inputs a recommendation.
@app.route('/recommend', methods=['POST'])
def recommend():
    book_title = request.form['book_title']
    recommendations = get_recommendations(book_title)
    return render_template('recommendations.html', recommendations=recommendations)

#Starts web server, and web application becomes accesible in the specified port.
if __name__ == '__main__':
    app.run(port=4996)