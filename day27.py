import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
data = {
    'News': [
        "Scientists discover a new Earth-like planet in the galaxy", # Real
        "SHOCKING! Aliens landed in New York and are eating pizza",  # Fake
        "Government passes new budget law for education",            # Real
        "Drink this secret magical water to cure all diseases 100%", # Fake
        "Stock market reaches all-time high this month"              # Real
    ],
    'Label': [1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)
ai_journalist = Pipeline([
    ('tf_idf_vectorizer', TfidfVectorizer(stop_words='english')),
    ('classifier', LogisticRegression())
])
print("AI is analyzing the news patterns... 🕵️‍♂️")
ai_journalist.fit(df['News'], df['Label'])
breaking_news = ["Viral secret to become a millionaire in 1 hour! Click here!"]
prediction = ai_journalist.predict(breaking_news)
print(f"\n--- 🚨 BREAKING NEWS CHECK ---")
print(f"Headline: '{breaking_news[0]}'")
if prediction[0] == 1:
    print("AI Verdict: This seems like a REAL news article. ✅")
else:
    print("AI Verdict: FAKE NEWS DETECTED! Don't trust this. ❌")
    
# Importing Pandas for our data, and our NLP tools for text math and classification
# Setting up our dataset: A mix of factual headlines and sensational clickbait
# The answers: 1 means REAL news, 0 means FAKE news
# Converting the raw dictionary into a neat Pandas DataFrame
# Building our AI Journalist Pipeline:
# Step 1: TfidfVectorizer translates text to math, AND ignores filler words (stop_words='english')
# Step 2: Logistic Regression learns to separate the 'real' math patterns from the 'fake' ones
# Training the AI on our 5 known headlines so it learns what clickbait looks like
# Creating a brand new, highly suspicious headline to test the AI
# The pipeline automatically strips the filler words, translates the rest, and makes a prediction
# --- Displaying the Results ---
# Printing the original headline for context
# If the AI outputs a 1, it prints the green 'REAL' news message
# If the AI outputs a 0, it triggers the red 'FAKE NEWS' alert!