import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def URL_transform(URL):
    URL = URL.lower()
    URL = nltk.word_tokenize(URL)
    
    y=[]
    for i in URL:
        if i.isalnum():
            y.append(i)
            
    URL = y[:]
    y.clear()

    for i in URL:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    URL = y[:]
    y.clear()
    
    for i in URL:
        y.append(ps.stem(i))
    
    return " ".join(y)

def main():
    st.title('PHISHING WEBSITE DETECTION')

    input_url = st.text_area('Enter the URL')

    if st.button('Predict'):
        # Preprocess
        transformed_URL = URL_transform(input_url)
        # Load vectorizer
        tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
        # Vectorize input
        vector_input = tfidf.transform([transformed_URL])
        # Load model
        model = pickle.load(open('model.pkl', 'rb'))
        # Predict
        result = model.predict(vector_input)[0]
        # Display result
        if result == 1:
            st.header("GOOD")
        else:
            st.header("BAD")

if __name__ == '__main__':
    main()
