import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
reviews = np.array(["Super movie and awesome", "Worst film very bad", "Good acting loved it", "Terrible story boring"])
sentiments = np.array([1, 0, 1, 0])
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('ai', MultinomialNB())
])
model.fit(reviews, sentiments)
st.title("🎬 AI Movie Review Analyzer")
st.write("Type a review below and my AI will guess if it's Positive or Negative!")
user_input = st.text_input("Enter your movie review:")
if st.button("Predict Sentiment"):
    if user_input:
        prediction = model.predict([user_input])
        
        
        if prediction[0] == 1:
            st.success("AI Says: This is a POSITIVE Review! 🤩")
            st.balloons() 
        else:
            st.error("AI Says: This is a NEGATIVE Review! 😡")
    else:
        st.warning("Please type a review first!")
        
# Importing streamlit (st), an amazing tool that turns Python scripts into interactive web pages
# Importing our NLP translation and brain tools from yesterday
# Setting up the raw training data: 4 sample movie reviews and their positive/negative labels
# Building and training the NLP Pipeline so the AI is fully educated before the user arrives
# --- STREAMLIT WEB APP UI STARTS HERE ---
# Creating a big, bold title for the top of our web page
# Adding a simple text description so the user knows what to do
# Creating an input box on the webpage where the user can type their custom review
# Creating a button. The code indented below this will ONLY run when the user clicks it!
# If the user actually typed something, feed their text to the trained AI pipeline
# If the AI predicts a 1 (Positive), display a nice green 'success' box on the screen