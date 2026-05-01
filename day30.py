import streamlit as st
from transformers import pipeline
st.set_page_config(page_title="My Own ChatGPT", page_icon="🤖")
st.title("🤖 My Personal AI Assistant")
st.write("Welcome! This AI is running entirely on your Python code.")
@st.cache_resource
def load_model():
    return pipeline('text-generation', model='gpt2')
ai_brain = load_model()
if "messages" not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
if user_input := st.chat_input("Ask me anything or give a starting phrase..."):
    
    with st.chat_message("user"):
        st.write(user_input)
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("assistant"):
        with st.spinner("AI is thinking... 🧠"):
            
            result = ai_brain(user_input, max_length=100, num_return_sequences=1, truncation=True)
            ai_reply = result[0]['generated_text']
            
            st.write(ai_reply)
            
    st.session_state.messages.append({"role": "assistant", "content": ai_reply})
    

# Importing Streamlit for the web interface and Transformers for the Generative AI brain
# Configuring the web page title and icon that appears in the browser tab
# Creating the main header and a welcome message for your personal assistant
# Using a 'decorator' to cache the model. This ensures the heavy AI brain only loads ONCE, making the app much faster!
# Defining the function to download and initialize the GPT-2 text generation model
# Initializing the actual AI brain by calling our cached function
# Setting up 'Session State'—this is the AI's memory. It stores your conversation so the chat history doesn't disappear when the page refreshes.
# Loop through all previous messages in the memory and display them in the chat window
# Creating the interactive chat input box. The ':=' (Walrus operator) saves your typing into 'user_input' instantly.
# Displaying your message in a 'user' bubble on the screen
# Saving your message into the AI's memory (Session State)
# Opening an 'assistant' bubble and showing a loading spinner while the AI thinks