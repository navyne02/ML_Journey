import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
cv = CountVectorizer()
word_matrix = cv.fit_transform(df['Tags'])
similarity = cosine_similarity(word_matrix)
def recommend_movie(movie_name):
    print(f"\n🍿 Because you watched '{movie_name}':")
    index = df[df['Movie'] == movie_name].index[0]
    distances = similarity[index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:3]
    for i in movies_list:
        print(f"👉 {df.iloc[i[0]].Movie}")
recommend_movie('Leo')

# Importing Pandas for our data table, and our Scikit-Learn tools for text math and similarity
# Setting up our movie database: Titles and their descriptive 'Tags' (genre, music director, director)
# Converting the raw dictionary into a neat Pandas DataFrame
# Initializing the text translator (CountVectorizer)
# Scanning all the 'Tags' and converting them into a matrix of numbers based on word counts
# The Magic Step: Calculating the 'Cosine Similarity' (the mathematical overlap) between every movie
# Creating a custom function that takes a movie you like and finds the closest matches
# Step 1: Finding the exact row number (index) of the movie you typed in
# Step 2: Grabbing that specific movie's similarity scores against all other movies in the database
# Step 3: Sorting the list by highest similarity score, grabbing the top 2 (skipping index 0, which is the movie itself!)
# Step 4: Looping through those top matches and printing their titles
# Running our function to see what the AI recommends if we liked 'Leo'!