import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Namma Data (Padangalum athoda Tags-um)
data = {
    'Movie': ['Leo', 'Vikram', 'Premam', 'Hridayam', 'Kaithi'],
    'Tags': [
        'action thriller anirudh loki', 
        'action thriller anirudh kamal', 
        'romance love college', 
        'romance love music', 
        'action thriller dark loki'
    ]
}
df = pd.DataFrame(data)

# 2. Text to Numbers (Day 19 trick!)
cv = CountVectorizer()
word_matrix = cv.fit_transform(df['Tags'])

# 3. AI Math Magic: Kandupudi Similarity! (0 to 1 kulla score varum)
similarity = cosine_similarity(word_matrix)

# 4. Recommendation Function (Oru padatha sonna, adutha 2 padatha sollum)
def recommend_movie(movie_name):
    print(f"\n🍿 Because you watched '{movie_name}':")
    
    # Padathoda Index-ah kandupudikka
    index = df[df['Movie'] == movie_name].index[0]
    
    # Antha padathukku matha padangaloda similarity scores edukkuthu
    distances = similarity[index]
    
    # High score la irunthu low score ku sort panrathu
    # [1:3] na first antha padame varum, so atha vittutu adutha 2 padatha edukkuthu
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:3]
    
    # Print the names
    for i in movies_list:
        print(f"👉 {df.iloc[i[0]].Movie}")

# --- Test it Here! ---
# Neenga 'Premam' illa 'Vikram' nu mathi run panni paarunga!
recommend_movie('Leo')