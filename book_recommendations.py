from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Your recommendation code here

# Cargar el conjunto de datos
df = pd.read_csv('books.csv') #data frame: es un objeto que obtiene todo el archivo

# Preprocesamiento de datos
# Crean una mega matriz en la que cada palabra esta representada por un numero, cuyo peso depende de la frecuencia, exceptuando las stop words.
tfidf_vectorizer = TfidfVectorizer(stop_words='english') #Palabras frecuentes sin importancia del idioma inglés (stop words)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['description'].fillna('')) # Convierte las palabras del .csv en números.unicamente usa el campo descripción.

# Calcular similitudes entre libros usando el producto interno del coseno
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix) #Lista que contiene las distancias entre descripciones de libros (de la matriz). Distancias entre todos los alementos

print(type(cosine_similarities))
# Función para obtener las recomendaciones de libros
# Inputeas el título del libro + numero de recomendaciones que tienes
def get_recommendations(book_title, num_recommendations=5):
    book_index = df.index[df['title'] == book_title].tolist() #Encuentra al libro en base a su posición de la lista. 
    if not book_index:
        return [] #Te devuelve una lista de recumenadción vacía.
    
    book_index = book_index[0]
    
    # Obtener las puntuaciones de similitud de este libro con otros libros
    sim_scores = list(enumerate(cosine_similarities[book_index])) #Crea una lista con todas las distancias entre nuestro libro y los del df.
    
    # Ordenar los libros por similitud
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True) #Ordena la lista de distancias de mayor a menor.
    
    # Obtener las puntuaciones de similitud de los libros recomendados
    recommendations = sim_scores[1:num_recommendations+1] #Crea una lista de recomendaciones (numeros) con los primeros 5 elementos de la lista de distancias.
    
    # Obtener los títulos de los libros recomendados
    recommended_books = [(df['title'][rec[0]], rec[1]) for rec in recommendations] #Convierte los numeros de recommendations que representnas las recomendaciones a strings y los mete en una lista
    
    return recommended_books #lista con tuplas (titulo + distancia)

# Ejemplo de recomendación
#book_title = "Pride and Prejudice"
#recommendations = get_recommendations(book_title)


app = Flask(__name__, template_folder='./template')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    book_title = request.form['book_title']
    recommendations = get_recommendations(book_title)
    return render_template('recommendations.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(port=4996)