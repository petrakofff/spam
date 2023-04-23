import streamlit as st
import requests
import pickle
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

# URL вашего файла модели на GitHub
url = "https://raw.githubusercontent.com/petrakofff/spam/main/models/myfile.pkl"

# Загрузка модели
response = requests.get(url)
model = pickle.loads(response.content)

# Функция предобработки данных
def prepare_data(url):
    tokenizer = RegexpTokenizer(r'[A-Za-z]+')
    stemmer = SnowballStemmer("english")
    cv = CountVectorizer()
    text_tokenized = tokenizer.tokenize(url)
    text_stemmed = [stemmer.stem(word) for word in text_tokenized]
    text_sent = ' '.join(text_stemmed)
    features = cv.fit_transform([text_sent])
    return features

# Заголовок приложения
st.title('Спам-детектор')

# Поле для ввода ссылки
url = st.text_input('Введите URL для проверки:')

# Кнопка для проверки
if st.button('Проверить'):
    # Предобработка данных
    features = prepare_data(url)
    # Применение модели для предсказания
    prediction = model.predict(features)
    # Вывод результата
    if prediction[0] == 1:
        st.write('Это спам!')
    else:
        st.write('Это не спам.')
