import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# 1. Namma Data (Real vs Fake News)
data = {
    'News': [
        "Scientists discover a new Earth-like planet in the galaxy", # Real
        "SHOCKING! Aliens landed in New York and are eating pizza",  # Fake
        "Government passes new budget law for education",            # Real
        "Drink this secret magical water to cure all diseases 100%", # Fake
        "Stock market reaches all-time high this month"              # Real
    ],
    'Label': [1, 0, 1, 0, 1]  # 1 = Real News ✅, 0 = Fake News ❌
}
df = pd.DataFrame(data)

# 2. Creating the Advanced NLP Pipeline
# TfidfVectorizer: Common words-ah ignore pannum (stop_words='english')
# LogisticRegression: Namma AI brain
ai_journalist = Pipeline([
    ('tf_idf_vectorizer', TfidfVectorizer(stop_words='english')),
    ('classifier', LogisticRegression())
])

# 3. Training the AI
print("AI is analyzing the news patterns... 🕵️‍♂️")
ai_journalist.fit(df['News'], df['Label'])

# 4. Test with a Breaking News!
breaking_news = ["Viral secret to become a millionaire in 1 hour! Click here!"]

prediction = ai_journalist.predict(breaking_news)

print(f"\n--- 🚨 BREAKING NEWS CHECK ---")
print(f"Headline: '{breaking_news[0]}'")

if prediction[0] == 1:
    print("AI Verdict: This seems like a REAL news article. ✅")
else:
    print("AI Verdict: FAKE NEWS DETECTED! Don't trust this. ❌")