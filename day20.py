import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np

# 1. Background Work: Train the AI (Like Day 19)
reviews = np.array(["Super movie and awesome", "Worst film very bad", "Good acting loved it", "Terrible story boring"])
sentiments = np.array([1, 0, 1, 0])

model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('ai', MultinomialNB())
])
model.fit(reviews, sentiments)

# ==========================================
# 2. FRONTEND WORK: Building the Web App UI
# ==========================================

# Website Title
st.title("🎬 AI Movie Review Analyzer")
st.write("Type a review below and my AI will guess if it's Positive or Negative!")

# Text Input Box for the user
user_input = st.text_input("Enter your movie review:")

# A Button to run the AI
if st.button("Predict Sentiment"):
    if user_input: # Check if box is not empty
        prediction = model.predict([user_input])
        
        # Display Results
        if prediction[0] == 1:
            st.success("AI Says: This is a POSITIVE Review! 🤩")
            st.balloons() # Streamlit magic! Shows balloons on screen
        else:
            st.error("AI Says: This is a NEGATIVE Review! 😡")
    else:
        st.warning("Please type a review first!")