import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Load vectorizer and model
vectorizer = joblib.load('tfidf_vectorizer.pkl')
model = joblib.load('model.pkl')

# Text cleaning function
def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    tokens = text.split()
    ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(ps.stem(word)) for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Streamlit UI
st.title("📰 Fake News Detection")

input_text = st.text_area("Enter news article text here")

if st.button("Check"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        clean_text = preprocess(input_text)
        vector_input = vectorizer.transform([clean_text]).toarray()
        prediction = model.predict(vector_input)[0]
        if prediction == 1:
            st.success("✅ This news looks **REAL**.")
        else:
            st.error("🚫 This news appears to be **FAKE**.")
