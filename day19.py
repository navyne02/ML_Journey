from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
reviews = np.array([
    "This movie is super and awesome", 
    "Worst movie very bad acting",      
    "Good story, I am very happy",     
    "Boring and terrible waste of time" 
])
sentiments = np.array([1, 0, 1, 0])
nlp_model = Pipeline([
    ('word_to_number', CountVectorizer()),
    ('ai_brain', MultinomialNB())
])
print("AI is reading the reviews... 📖")
nlp_model.fit(reviews, sentiments)
new_review = ["The movie is really good and super"]
prediction = nlp_model.predict(new_review)
print("\n--- AI Movie Reviewer ---")
print(f"Input Review: '{new_review[0]}'")
if prediction[0] == 1:
    print("AI Says: This is a POSITIVE review! 🤩")
else:
    print("AI Says: This is a NEGATIVE review! 😡")
    
# Importing CountVectorizer (to translate text to math) and MultinomialNB (an AI that excels at reading)
# Importing the Pipeline tool to seamlessly connect our translation and learning steps
# Setting up our raw text data: An array of 4 short movie reviews
# The sentiment labels for those reviews: 1 means POSITIVE, 0 means NEGATIVE
# Building our NLP Assembly Line (Natural Language Processing)
# Step 1: 'word_to_number' scans the text, counts the words, and turns the sentences into a matrix of numbers
# Step 2: The 'ai_brain' (Naive Bayes) looks at those numbers to learn which words imply positive vs negative feelings
# Training the model: It reads the text, builds a dictionary, counts the words, and learns the sentiment all at once!
# Giving the AI a brand new movie review it has never seen before
# The pipeline automatically translates this new English text into numbers and predicts the sentiment
# Printing a fun, dynamic message depending on whether the AI thought it was a POSITIVE or NEGATIVE review