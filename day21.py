import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
data = {
    'Restaurant_Name': ['Spice Garden', 'Namma Veetu Saapadu', 'Pasta Bar', 'Dragon Wok', 'Taco Fiesta', 'Pizza Corner'],
    'Cuisine': ['Indian', 'Indian', 'Italian', 'Chinese', 'Mexican', 'Italian'],
    'Ingredients': [
        'garlic, ginger, turmeric, chicken, masala', 
        'rice, sambar, curry leaves, mustard, tomato',
        'tomato, cheese, basil, pasta, olive oil', 
        'soy sauce, garlic, noodles, pork, chili', 
        'corn, beans, cheese, salsa, tortilla', 
        'dough, tomato, mozzarella, oregano'
    ],
    'Rating': [4.5, 4.8, 4.2, 4.0, 4.3, 4.1],
    'Cost_for_Two': [500, 300, 800, 600, 400, 700]
}
df = pd.DataFrame(data)
ai_model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
])
ai_model.fit(df['Ingredients'], df['Cuisine'])
st.set_page_config(page_title="Smart Cuisine AI", page_icon="🍽️")
st.title("🍽️ Smart Cuisine & Restaurant Finder")
st.write("Veetla irukkura ingredients-ah type pannunga, namma AI athu entha Cuisine nu kandupudichu, athuketha Restaurants-ah suggest pannum!")
st.markdown("---")
st.subheader("👨‍🍳 What's in your fridge?")
user_ingredients = st.text_input("Example: tomato, cheese, basil")
if st.button("Predict Cuisine & Find Restaurants"):
    if user_ingredients:
        predicted_cuisine = ai_model.predict([user_ingredients])[0]
        st.success(f"### 🪄 AI Predicts: **{predicted_cuisine} Cuisine**")
        
        st.markdown("---")
        
        st.subheader(f"🏆 Top {predicted_cuisine} Restaurants for you:")
        
        
        recommended_df = df[df['Cuisine'] == predicted_cuisine].sort_values(by='Rating', ascending=False)
        
        if not recommended_df.empty:
        
            for index, row in recommended_df.iterrows():
                col1, col2, col3 = st.columns(3)
                col1.metric(label="Restaurant", value=row['Restaurant_Name'])
                col2.metric(label="Rating", value=f"{row['Rating']} ⭐")
                col3.metric(label="Cost for Two", value=f"₹{row['Cost_for_Two']}")
                st.write("") 
        else:
            st.info("No restaurants found for this cuisine in our current database.")
            
    else:
        st.warning("Please enter some ingredients first!")
        
# Importing Streamlit for the UI, Pandas for handling data tables, and our ML tools
# Setting up our raw restaurant dataset with names, cuisines, ingredients, ratings, and costs
# Converting that raw dictionary into a clean Pandas DataFrame (like an invisible Excel table)
# Building our AI Pipeline: 
# Step 1: CountVectorizer translates the English ingredient lists into math/number counts
# Step 2: RandomForest learns which ingredient combinations belong to which cuisine
# Training the AI pipeline using ONLY the 'Ingredients' column to predict the 'Cuisine' column
# --- STREAMLIT WEB APP UI ---
# Configuring the web page's browser tab title and icon
# Creating the main title and the descriptive text for the user
# Creating a text input box asking the user what ingredients they have in their fridge
# Creating the prediction button. The AI will only run when this is clicked!
# If the user typed something, feed their text to the trained AI pipeline
# The AI outputs a prediction array; we grab the first item [0] to get the predicted cuisine string