from flask import Flask, request, render_template
import gensim.downloader as api
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

app = Flask(__name__)

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load models
word2vec_model = api.load("word2vec-google-news-300")
with open('random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

def preprocess_text(text):
    # Remove URLs, numbers, and special characters
    text = re.sub(r'http\S+', '', text)  # URLs
    text = re.sub(r'\d+', '', text)       # Numbers
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)  # Emails
    
    # Convert to lowercase and split into words
    words = text.lower().split()

    # Remove stopwords and short words (<3 letters)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

    return ' '.join(lemmatized_words)

def get_word2vec_vector(text, model):
    # Convert preprocessed text to vectors
    vectors = []
    for word in text.split():
        if word in model.key_to_index:  # Check if word exists in vocabulary
            vectors.append(model[word])
    
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

def predict_label(text, model, word2vec_model, label_encoder=None):
    # Preprocess the text
    cleaned_text = preprocess_text(text)
    
    # Convert to Word2Vec vector
    text_vector = get_word2vec_vector(cleaned_text, word2vec_model)
    
    # Reshape for model input
    text_vector = text_vector.reshape(1, -1)
    
    # Predict
    prediction = model.predict(text_vector)[0]
    
    # Decode label
    if label_encoder:
        prediction = label_encoder.inverse_transform([prediction])[0]
    
    return prediction

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        text = request.form['text']
        prediction = predict_label(text, rf_model, word2vec_model, label_encoder)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)