import streamlit as st
import pickle
import string
import nltk
import os
nltk_data_path = os.path.join(os.path.dirname(__file__), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)

def transformation(text):
    text = text.lower()    #lower case
    text = nltk.word_tokenize(text)     #tokenize

    l = []     #removing special char
    for i in text:
        if i.isalnum():
            l.append(i)
    text= l[:]
    l.clear()
    for i in text:
        if i not in stop_words and i not in punctuations:
            l.append(i)
    text=l[:]
    l.clear()
    for i in text:
        l.append(ps.stem(i))
    return " ".join(l)

tfdif = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title('SMS spam Classifier')

input_sms = st.text_area("Enter your message")

if st.button('Predict'):
    #preprocess
    transformed_sms = transformation(input_sms)

    #vectorize
    vector_input = tfdif.transform([transformed_sms])

    #model
    result = model.predict(vector_input)[0]

    #result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
