import re
import string
from collections import Counter

import altair as alt
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.corpus import stopwords
from wordcloud import WordCloud

import streamlit as st

"""
altair==5.4.1
matplotlib==3.9.2
nltk==3.9.1
pandas==2.2.3
streamlit==1.39.0
wordcloud==1.9.3
"""

# Descargar stopwords de nltk si es necesario
# nltk.download('stopwords')


# Función para limpiar y procesar el texto con stopwords adicionales y longitud mínima de palabra
def clean_text(text, extra_stopwords, min_word_length):
    text = text.lower()  # Convertir a minúsculas
    text = "".join(
        [char for char in text if char not in string.punctuation]
    )  # Quitar puntuación
    tokens = text.split()  # Tokenizar

    # Cargar stopwords en español y añadir las personalizadas
    stop_words = set(stopwords.words("spanish"))
    if extra_stopwords:
        stop_words.update(extra_stopwords)

    # Filtrar stopwords y palabras más cortas que min_word_length
    tokens = [
        word
        for word in tokens
        if word not in stop_words and len(word) >= min_word_length
    ]

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
    data = pd.read_csv("../streamlit/jokes_df.csv")  # Ruta del archivo
    return data


# Función para reiniciar el estado de los botones
def reset_buttons():
    st.session_state["show_df"] = False
    st.session_state["clean_df"] = False
    st.session_state["show_wordcloud"] = False
    st.session_state["show_top10"] = False


# Función para generar gráfico de barras con altair
def generate_altair_bar_chart(df_common_words):
    chart = (
        alt.Chart(df_common_words)
        .mark_bar()
        .encode(
            x=alt.X("Palabra:N", sort="-y", title="Palabra"),
            y=alt.Y("Frecuencia:Q", title="Frecuencia"),
            tooltip=["Palabra", "Frecuencia"],
        )
        .properties(width=600, height=400)
        .configure_axis(labelFontSize=12)
    )

    st.altair_chart(chart, use_container_width=True)


# Función principal de la app
def main():
    st.title("Análisis de chistes por festival y humorista")

    # Cargar los datos
    data = load_data()

    # Inicializar las listas de eventos y artistas (shows)
    event_names = data["edicion"].unique()
    show_names = data["show_name"].unique()

    # Inicializar el estado de los botones si no existen
    if "show_df" not in st.session_state:
        reset_buttons()

    # Selección de filtros
    selected_event = st.selectbox(
        "Selecciona un festival", ["Todos"] + list(event_names)
    )
    if selected_event != "Todos":
        available_shows = data[data["edicion"] == selected_event]["show_name"].unique()
    else:
        available_shows = show_names

    selected_show = st.selectbox(
        "Selecciona un humorista", ["Todos"] + list(available_shows)
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

    # Agregar cuadro de texto para nuevas stopwords
    custom_stopwords = st.text_input(
        "Escribe nuevas stopwords separadas por comas"
    ).split(",")
    custom_stopwords = [
        word.strip() for word in custom_stopwords if word
    ]  # Limpiar input

    # Agregar slider para la longitud mínima de palabras
    min_word_length = st.slider(
        "Selecciona la longitud mínima de las palabras a considerar", 1, 10, 3
    )

    # Si se cambia algún parámetro (evento, show, stopwords, longitud de palabras), resetear los botones
    if st.button("Aplicar cambios"):
        reset_buttons()

    # Mostrar el dataset original
    if not st.session_state["show_df"]:
        if st.button("Mostrar datos originales"):
            st.session_state["show_df"] = True

    # Mostrar el dataset filtrado (si ya se activó el botón correspondiente)
    if st.session_state["show_df"]:
        st.subheader("Dataset original sin limpiar")
        st.write(filtered_data)

    # Limpiar y procesar el texto de los chistes filtrados
    if st.session_state["show_df"] and not st.session_state["clean_df"]:
        if st.button("Limpiar datos"):
            st.session_state["clean_df"] = True
            st.session_state["chistes_limpios"] = filtered_data["text"].apply(
                lambda x: clean_text(x, custom_stopwords, min_word_length)
            )

    # Mostrar el dataset limpio
    if st.session_state["clean_df"]:
        st.subheader("Dataset sin stopwords y signos de puntuación")
        st.write(st.session_state["chistes_limpios"])

    # Generar y mostrar wordcloud
    if st.session_state["clean_df"] and not st.session_state["show_wordcloud"]:
        if st.button("Mostrar wordcloud"):
            st.session_state["show_wordcloud"] = True

    if st.session_state["show_wordcloud"]:
        st.subheader(f"Wordcloud de chistes")
        if st.session_state["chistes_limpios"] is not None:
            all_words = [
                word
                for chiste in st.session_state["chistes_limpios"]
                for word in chiste
            ]
            if all_words:
                generate_wordcloud(all_words)
            else:
                st.write("No hay chistes disponibles para este filtro.")

    # Generar y mostrar el gráfico de las 10 palabras más comunes
    if st.session_state["show_wordcloud"] and not st.session_state["show_top10"]:
        if st.button("Top 10 palabras"):
            st.session_state["show_top10"] = True

    if st.session_state["show_top10"]:
        st.subheader("Top 10 palabras más frecuentes")
        if st.session_state["chistes_limpios"] is not None:
            all_words = [
                word
                for chiste in st.session_state["chistes_limpios"]
                for word in chiste
            ]
            word_counts = Counter(all_words)
            common_words = word_counts.most_common(10)
            if common_words:
                # Crear DataFrame y ordenar por frecuencia
                df_common_words = pd.DataFrame(
                    common_words, columns=["Palabra", "Frecuencia"]
                )

                # Generar gráfico usando Altair
                generate_altair_bar_chart(df_common_words)
            else:
                st.write("No hay palabras suficientes para mostrar el conteo.")


if __name__ == "__main__":
    main()
