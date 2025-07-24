import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
import base64

# ---------- Function to Encode Image to Base64 ----------
def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ---------- Load Avatars ----------
user_avatar = get_base64_image("static/user.jpg")  # Make sure these images are in the same folder
bot_avatar = get_base64_image("static/bot.png")

# ---------- Page Config ----------
st.set_page_config(page_title="Loan Assistant", layout="centered")

# ---------- Custom CSS ----------
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        background-color: #f0f4f8;
    }}

    .chat-title {{
        font-size: 2.8rem;
        font-weight: 800;
        color: #2563EB;
        text-align: center;
        margin-top: 10px;
        animation: fadeInDown 1s ease;
    }}

    .chat-subtitle {{
        font-size: 1rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        animation: fadeInDown 1.2s ease;
    }}

    .chat-container {{
        background-color: #ffffff;
        max-width: 750px;
        margin: 0 auto;
        padding: 1rem;
        border-radius: 20px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.05);
        height: 70vh;
        overflow-y: auto;
        display: flex;
        flex-direction: column-reverse;
        height: 100%;
    }}


    .chat-bubble {{
        display: flex;
        align-items: flex-start;
        margin-bottom: 1.2rem;
        animation: fadeInUp 0.5s ease;
    }}

    .chat-bubble.bot {{
        flex-direction: row;
    }}

    .chat-bubble.user {{
        flex-direction: row-reverse;
    }}

    .bubble-content {{
        max-width: 75%;
        padding: 0.9rem 1.2rem;
        border-radius: 16px;
        background-color: #E0F2FE;
        color: #111;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }}

    .chat-bubble.bot .bubble-content {{
        background-color: #F3F4F6;
        color: #333;
    }}

    .avatar {{
        width: 38px;
        height: 38px;
        border-radius: 50%;
        margin: 0 12px;
    }}

    @keyframes fadeInDown {{
        from {{ opacity: 0; transform: translateY(-20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .stTextInput > div > div > input {{
        background-color: #fff;
        border-radius: 12px;
        padding: 12px;
        border: 1px solid #ccc;
    }}

    .stButton > button {{
        background-color: #2563EB;
        color: white;
        font-weight: 600;
        padding: 0.6rem 1.4rem;
        border-radius: 12px;
        border: none;
    }}
    </style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.markdown('<div class="chat-title">Loan Approval Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="chat-subtitle">Ask anything about loan eligibility or the application process.</div>', unsafe_allow_html=True)

# ---------- Model Setup ----------
model = OllamaLLM(model="tinyllama")
template = """
You are an expert in answering questions about a loan approval system.
Here are some relevant reviews: {reviews}
Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# ---------- Session State ----------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- Input Form ----------
with st.form("chat_form", clear_on_submit=True):
    question = st.text_input("💬 Type your question below:", placeholder="e.g., Why was my loan application rejected?")
    submitted = st.form_submit_button("Send")

# ---------- Handle Response ----------
if submitted and question:
    with st.spinner("🤖 Bot is typing..."):
        reviews = retriever.invoke(question)
        answer = chain.invoke({"reviews": reviews, "question": question})
        st.session_state.history.append((question, answer))

# ---------- Chat Display ----------
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for user_msg, bot_msg in reversed(st.session_state.history):
    st.markdown(f"""
        <div class="chat-bubble user">
            <img class="avatar" src="data:image/png;base64,{user_avatar}" alt="user">
            <div class="bubble-content">{user_msg}</div>
        </div>
        <div class="chat-bubble bot">
            <img class="avatar" src="data:image/png;base64,{bot_avatar}" alt="bot">
            <div class="bubble-content">{bot_msg}</div>
        </div>
    """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
