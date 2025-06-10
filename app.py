import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Konfigurasi halaman
st.set_page_config(page_title="Dashboard Sentimen TIX ID", layout="wide")

# Load model & vectorizer
model = joblib.load("svm_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Load dataset
df = pd.read_csv("TIX-ID_2023-2025_Bersih.csv")

# Validasi kolom
if 'clean_content' not in df.columns:
    st.error("Kolom 'clean_content' tidak ditemukan di dataset.")
    st.stop()

# Tambah kolom sentimen jika belum ada
if 'sentiment' not in df.columns:
    df['sentiment'] = df['score'].apply(lambda x: 'negatif' if x <= 2 else 'netral' if x == 3 else 'positif')

# Sidebar navigasi
st.sidebar.title("📊 Navigasi Dashboard")
selected = st.sidebar.radio("Menu", ["Home", "WordCloud", "Tabel Review"])

# Ringkasan data
if selected == "Home":
    st.title("📊 Ringkasan Sentimen Ulasan TIX ID")

    sentiment_counts = df['sentiment'].value_counts().reindex(['positif', 'netral', 'negatif'])
    total = sentiment_counts.sum()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribusi Sentimen")
        fig, ax = plt.subplots()
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="Set2", ax=ax)
        ax.set_ylabel("Jumlah")
        ax.set_xlabel("Sentimen")
        st.pyplot(fig)

    with col2:
        st.subheader("Komposisi Sentimen")
        fig2, ax2 = plt.subplots()
        ax2.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90,
                colors=sns.color_palette("Set2"))
        ax2.axis('equal')
        st.pyplot(fig2)

    st.markdown(f"**Total Data:** {total}")
    st.markdown(f"""
    - 👍 **Positif**: {sentiment_counts['positif']}
    - 😐 **Netral**: {sentiment_counts['netral']}
    - 👎 **Negatif**: {sentiment_counts['negatif']}
    """)

# WordCloud
elif selected == "WordCloud":
    st.title("☁️ WordCloud per Sentimen")
    sentimen = st.radio("Pilih sentimen:", ["positif", "netral", "negatif"], horizontal=True)
    teks = " ".join(df[df['sentiment'] == sentimen]['clean_content'].dropna())
    cmap = {"positif": "Greens", "netral": "gray", "negatif": "Reds"}[sentimen]

    if teks.strip():
        wc = WordCloud(width=800, height=400, background_color='white', colormap=cmap).generate(teks)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wc, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)
    else:
        st.warning("Tidak ada data untuk sentimen ini.")

# Tabel review
elif selected == "Tabel Review":
    st.title("📄 Tabel Ulasan")
    jumlah = st.slider("Tampilkan berapa ulasan?", 5, 100, 10)
    st.dataframe(df[['clean_content', 'sentiment']].rename(columns={'clean_content': 'Ulasan'}).head(jumlah),
                 use_container_width=True)
