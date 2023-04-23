import streamlit as st
import pickle
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

# Загрузка обученной модели из файла
with open('LR.pkl', 'rb') as f:
    model = pickle.load(f)

# Определение функции для классификации ссылки
def classify_link(link):
    tokenizer = RegexpTokenizer(r'[A-Za-z]+')
    stemmer = SnowballStemmer("english")
    cv = CountVectorizer()

    link_tokenized = tokenizer.tokenize(link)
    link_stemmed = [stemmer.stem(word) for word in link_tokenized]
    link_sent = ' '.join(link_stemmed)

    features = cv.fit_transform([link_sent])
    prediction = model.predict(features)
    if prediction[0] == 1:
        return "Спам"
    else:
        return "Не спам"

# Создание веб-страницы с помощью Streamlit
st.title("Классификация ссылок на спам")

# Получение ссылки от пользователя
link = st.text_input("Введите ссылку для классификации")

# Обработка ссылки и вывод результата
if link:
    classification = classify_link(link)
    st.write("Результат классификации:", classification)
