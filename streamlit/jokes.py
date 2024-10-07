import re
import string
from collections import Counter

import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.corpus import stopwords
from wordcloud import WordCloud

import streamlit as st

# Descargar stopwords de nltk si es necesario
# nltk.download('stopwords')


# Función para limpiar y procesar el texto
def clean_text(text):
    text = text.lower()  # Convertir a minúsculas
    text = "".join(
        [char for char in text if char not in string.punctuation]
    )  # Quitar puntuación
    tokens = text.split()  # Tokenizar
    stop_words = set(stopwords.words("spanish"))  # Filtrar stopwords en español
    tokens = [word for word in tokens if word not in stop_words]
    return tokens


# Función para generar wordcloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
        " ".join(text)
    )
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)


# Cargar el dataset
@st.cache_data
def load_data():
    # Cargar el dataset desde archivo CSV o formato adecuado
    # Asegúrate de cambiar la ruta del archivo a la que corresponda
    data = pd.read_csv("streamlit/jokes_df.csv")
    return data


# Función principal de la app
def main():
    st.title("Análisis de Chistes por Evento y Show")

    # Inicializar las listas de eventos y artistas (shows)
    event_names = data["edicion"].unique()
    show_names = data["show_name"].unique()

    # Selección de filtros
    selected_event = st.selectbox("Selecciona un Evento", ["Todos"] + list(event_names))

    # Si se selecciona un evento, filtrar los shows disponibles
    if selected_event != "Todos":
        available_shows = data[data["edicion"] == selected_event]["show_name"].unique()
    else:
        available_shows = show_names

    selected_show = st.selectbox(
        "Selecciona un Show", ["Todos"] + list(available_shows)
    )

    # Filtrar los datos según la selección
    if selected_event != "Todos" and selected_show != "Todos":
        filtered_data = data[
            (data["edicion"] == selected_event) & (data["show_name"] == selected_show)
        ]
    elif selected_event != "Todos":
        filtered_data = data[data["edicion"] == selected_event]
    elif selected_show != "Todos":
        filtered_data = data[data["show_name"] == selected_show]
    else:
        filtered_data = data

    # Mostrar el dataset filtrado
    st.subheader("Dataset Filtrado")
    st.write(filtered_data)

    # Limpiar y procesar el texto de los chistes filtrados
    chistes = filtered_data["text"].apply(clean_text)

    # Wordcloud basado en los datos filtrados
    st.subheader(f"Wordcloud de Chistes: {selected_event} - {selected_show}")
    all_words = [word for chiste in chistes for word in chiste]
    if all_words:  # Si hay palabras para mostrar
        generate_wordcloud(all_words)
    else:
        st.write("No hay chistes disponibles para este filtro.")

    # Conteo de palabras más comunes
    st.subheader("Palabras más frecuentes")
    word_counts = Counter(all_words)
    common_words = word_counts.most_common(10)
    if common_words:
        df_common_words = pd.DataFrame(common_words, columns=["Palabra", "Frecuencia"])
        st.bar_chart(df_common_words.set_index("Palabra"))
    else:
        st.write("No hay palabras suficientes para mostrar el conteo.")


if __name__ == "__main__":
    main()
