# Content-Based Filtering Experiment

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse import hstack

df = pd.read_csv('books.csv', nrows=1000)

tfidf_vectorizer_description = TfidfVectorizer(stop_words='english') 
tfidf_vectorizer_genre = TfidfVectorizer(stop_words='english')

tfidf_matrix_description = tfidf_vectorizer_description.fit_transform(df['description'].fillna('')) 
tfidf_matrix_genre = tfidf_vectorizer_genre.fit_transform(df['genres'].fillna(''))

tfidf_matrices = [tfidf_matrix_description, tfidf_matrix_genre]
tfidf_matrix_combined = hstack(tfidf_matrices)

cosine_similarities = linear_kernel(tfidf_matrix_combined, tfidf_matrix_combined)

def get_recommendations(book_title, num_recommendations=6):
    book_index = df.index[df['title'] == book_title].tolist() 
    if not book_index:
        return [] 
    
    book_index = book_index[0]
    
    sim_scores = list(enumerate(cosine_similarities[book_index])) 
    
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) 

    recommendations = sim_scores[1:num_recommendations+1]

    recommended_books = [(df['title'][rec[0]], min(100, round(rec[1] * 100))) for rec in recommendations]
 
    return recommended_books

# User provides a book title for recommendations
user_input_book_title = input("Please enter a book title: ")

content_based_recommendations = get_recommendations(user_input_book_title)

print("Content-Based Filtering Results:")
for book, similarity in content_based_recommendations:
    print(f"Book: {book}")
    print(f"Similarity: {similarity}%")