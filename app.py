import pickle
import streamlit as st

# Загрузка обученной модели из файла
with open('spam_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# Определение функции для классификации ссылки
def classify_link(link):
    prediction = model.predict([link])
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
