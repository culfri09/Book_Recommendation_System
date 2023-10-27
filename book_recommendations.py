from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse import hstack

'''
#---Recommendation Algorithm---
'''

# Loads data set into data frame.
df = pd.read_csv('books.csv') 

# Data preprocessing: Create TF-IDF matrices for description and genres represented by a number (the weight depends in the frequency). Stop words are not included.
tfidf_vectorizer_description = TfidfVectorizer(stop_words='english') #Initializes the TfidfVectorizer with stop words removed.
tfidf_vectorizer_genre = TfidfVectorizer(stop_words='english')

tfidf_matrix_description = tfidf_vectorizer_description.fit_transform(df['description'].fillna('')) #Fits and transform the description information from the df to create the matrix.
tfidf_matrix_genre = tfidf_vectorizer_genre.fit_transform(df['genres'].fillna('')) #Fits and transform the genres information from the df to create the matrix.

# Combine the TF-IDF matrices horizontally (side by side) to create a single feature matrix.
tfidf_matrices = [tfidf_matrix_description, tfidf_matrix_genre]
tfidf_matrix_combined = hstack(tfidf_matrices)


# Computes cosine similarities
cosine_similarities = linear_kernel(tfidf_matrix_combined, tfidf_matrix_combined) # List which contains the distance between ALL the book's descriptions within the matrix. 

# Method to obtain the book recommendations. Takes as parameters the book's name and the number of recommendations that will be given. 
def get_recommendations(book_title, num_recommendations=6):
    book_index = df.index[df['title'] == book_title].tolist() # Finds the index of the book within the list. 
    if not book_index:
        return [] #If title is not within the df, it returns an empty recommendation.
    
    book_index = book_index[0]
    
    # Obtains the similitude scores between the input book with the rest of the books in the df. 
    sim_scores = list(enumerate(cosine_similarities[book_index])) #Creates a list with all of the distances between the input book and the ones in the df.

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) # Sorts books based on similitude score (from highest to lowest).
    
    recommendations = sim_scores[1:num_recommendations+1] #Creates list with the first 6 items of the sim_scores (list with the sorted books by similitude)
    
    recommended_books = [(df['title'][rec[0]],df['author'][rec[0]], min(100, round(rec[1] * 100))) for rec in recommendations] #Transforms the numbers of the recommendations into strings and adds them to recommendation_books list
    return recommended_books 

def get_autocomplete_suggestions(query):
    # Filter book titles based on the user's query
    matching_titles = df[df['title'].str.contains(query, case=False, na=False)]['title'].tolist()

    # Limit the number of suggestions to a 10.
    suggestions = matching_titles[:10]  

    return suggestions

'''
#---Flask integration---
'''

app = Flask(__name__, template_folder='./template') #Creates a Flask application named app, and specifies path where the rendered HTML pages are located.

#Loads main page when user accesses the root of the URL.
@app.route('/')
def index():
    return render_template('index.html')

#Handles autocomplete requests based on user input.
@app.route('/autocomplete', methods=['POST'])
def autocomplete():
    query = request.form['query']
    suggestions = get_autocomplete_suggestions(query)
    return jsonify(suggestions)

#Handles book recommendation requests.
@app.route('/recommend', methods=['POST'])
def recommend():
    book_title = request.form['book_title'] #Retrieve the 'book_title' value from a form submission
    recommendations = get_recommendations(book_title) #Passes user input to fetch book recommendations.

    #If book is not found, loads book not found page.
    if not recommendations:
        return render_template('book_not_found.html')
    
    #Prepare the recommendations with color information based on similarity score.
    colored_recommendations = []
    for book, author, similarity in recommendations:
        #Determine a color based on the 'similarity' value.
        color = 'text-bg-success' if similarity >= 70 else ('text-bg-warning' if similarity >= 30 else 'text-bg-danger')
        #Create a dictionary for the book recommendation with book title, author, similarity, and color.
        colored_recommendations.append({'book': book, 'author': author, 'similarity': similarity, 'color': color})

    #Renders recommendation page and passes book title and colors as arguments. 
    return render_template('recommendations.html', recommendations=colored_recommendations, book_title=book_title)


#Starts web server, and web application becomes accesible in the specified port.
if __name__ == '__main__':
    app.run(port=4996)



