import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    stop_words = set(stopwords.words('indonesian'))
    tokens = word_tokenize(text.lower())
    filtered = [word for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered)

@st.cache_data
def load_data():
    df = pd.read_csv('data.csv')
    df['processed_question'] = df['question'].apply(preprocess_text)
    return df

df = load_data()
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['processed_question'])

# Header
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://via.placeholder.com/120x120.png?text=MTS+Logo", width=100)
with col2:
    st.title("🤖 Chatbot MTS Al-Hikmah")
    st.markdown("_Informasi Sekolah Berbasis NLP_")

# Inisialisasi session
if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Assalamu'alaikum warahmatullahi wabarakatuh 🙏\n\nSelamat datang! Saya siap jawab pertanyaan tentang MTS Al-Hikmah.\nPilih tombol di bawah atau langsung ketik."
    })

# Tampilkan riwayat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Tombol quick reply
st.markdown("**🔹 Pilih topik cepat:**")
col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)

quick_prompt = None
if col_btn1.button("🏫 Lokasi Sekolah"):
    quick_prompt = "dimana lokasi sekolah"
if col_btn2.button("💰 Biaya SPP"):
    quick_prompt = "biaya sekolah berapa"
if col_btn3.button("📝 Pendaftaran"):
    quick_prompt = "cara daftar siswa baru"
if col_btn4.button("🏆 Program Unggulan"):
    quick_prompt = "program unggulan"

# Chat input selalu ada
user_input = st.chat_input("Ketik pertanyaanmu di sini...")

prompt = quick_prompt or user_input

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    processed_prompt = preprocess_text(prompt)
    prompt_vector = vectorizer.transform([processed_prompt])
    similarities = cosine_similarity(prompt_vector, tfidf_matrix).flatten()
    best_match_index = similarities.argmax()
    best_similarity = similarities[best_match_index]

    if best_similarity > 0.3:
        response = df['answer'].iloc[best_match_index]
    else:
        response = "Maaf, saya belum tahu jawabannya 😔\nHubungi admin: (021) 786-1234 atau info@mtsalhikmah.sch.id"

    if any(k in prompt.lower() for k in ["lokasi", "alamat", "fasilitas"]):
        st.markdown("### 🖼️ Foto Sekolah")
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.image("https://via.placeholder.com/600x400.png?text=Gedung+Utama", caption="Gedung Utama")
        with img_col2:
            st.image("https://via.placeholder.com/600x400.png?text=Fasilitas+Sekolah", caption="Lapangan & Mushola")

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

st.markdown("---")
st.caption("Proyek Mata Kuliah NLP • Streamlit + TF-IDF")