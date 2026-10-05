import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="MIRA - Menstrual Health Chatbot", page_icon="🌸")
st.title("🌸 MIRA")
st.subheader("AI-Based Menstrual Health and Hygiene Awareness Chatbot")

@st.cache_data
def load_data():
    data = pd.read_csv("MENST.csv")
    data.columns = data.columns.str.strip()
    q_col = next((c for c in ["Question", "question", "Questions", "questions"] if c in data.columns), None)
    a_col = next((c for c in ["Answer", "answer", "Answers", "answers"] if c in data.columns), None)
    data = data.dropna(subset=[q_col, a_col]).copy()
    data[q_col] = data[q_col].astype(str)
    data[a_col] = data[a_col].astype(str)
    return data[(data[q_col].str.strip() != "") & (data[a_col].str.strip() != "")].reset_index(drop=True), q_col, a_col

try:
    data, q_col, a_col = load_data()
except Exception as e:
    st.error("Error loading MENST.csv dataset. Please verify the file is in your repository.")
    st.stop()

@st.cache_resource
def build_model(questions):
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(questions)
    return vectorizer, vectors

vectorizer, question_vectors = build_model(data[q_col])

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am MIRA, your menstrual health awareness chatbot."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Ask MIRA a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    user_vector = vectorizer.transform([prompt])
    similarity_scores = cosine_similarity(user_vector, question_vectors)
    best_match_index = similarity_scores.argmax()

    if similarity_scores[0, best_match_index] >= 0.20:
        answer = str(data.iloc[best_match_index][a_col])
    else:
        answer = "I'm sorry, I could not find a sufficiently relevant answer in my dataset. For medical concerns, please consult a qualified healthcare professional."

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)
