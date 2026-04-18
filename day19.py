from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np

# 1. Namma Data (Movie Reviews and Result)
reviews = np.array([
    "This movie is super and awesome",  # Positive
    "Worst movie very bad acting",      # Negative
    "Good story, I am very happy",      # Positive
    "Boring and terrible waste of time" # Negative
])

# 1 = Positive, 0 = Negative
sentiments = np.array([1, 0, 1, 0])

# 2. Creating the NLP Pipeline (Day 18-la padicha the same trick!)
# Step 1: Words-ah Numbers-ah mathu (CountVectorizer)
# Step 2: Naive Bayes algorithm use panni classify pannu (MultinomialNB)
nlp_model = Pipeline([
    ('word_to_number', CountVectorizer()),
    ('ai_brain', MultinomialNB())
])

# 3. Training the Text AI
print("AI is reading the reviews... 📖")
nlp_model.fit(reviews, sentiments)

# 4. Test with a New Review
# Neenga intha review-va mathi kooda try pannalam!
new_review = ["The movie is really good and super"]
prediction = nlp_model.predict(new_review)

print("\n--- AI Movie Reviewer ---")
print(f"Input Review: '{new_review[0]}'")

if prediction[0] == 1:
    print("AI Says: This is a POSITIVE review! 🤩")
else:
    print("AI Says: This is a NEGATIVE review! 😡")