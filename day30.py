import streamlit as st
from transformers import pipeline

# 1. UI Setup (The Body)
st.set_page_config(page_title="My Own ChatGPT", page_icon="🤖")
st.title("🤖 My Personal AI Assistant")
st.write("Welcome! This AI is running entirely on your Python code.")

# 2. Load the AI Model (The Brain)
# @st.cache_resource ngradhu romba mukkiyam. Ithu model-ah orey oru thadava 
# download/load panni memory-la vechukkum. Appo thaan app fast-ah irukkum.
@st.cache_resource
def load_model():
    return pipeline('text-generation', model='gpt2')

ai_brain = load_model()

# 3. Chat History Setup (Pazhaiya messages-ah niyabagam vechukka)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display old messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 4. The Chat Logic!
# User input box
if user_input := st.chat_input("Ask me anything or give a starting phrase..."):
    
    # Show user's message
    with st.chat_message("user"):
        st.write(user_input)
    # Save it to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # AI generates response
    with st.chat_message("assistant"):
        with st.spinner("AI is thinking... 🧠"):
            # Generative AI doing its magic
            result = ai_brain(user_input, max_length=100, num_return_sequences=1, truncation=True)
            ai_reply = result[0]['generated_text']
            
            st.write(ai_reply)
            
    # Save AI response to history
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})