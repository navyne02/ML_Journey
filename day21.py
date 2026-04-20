import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# ==========================================
# 1. DATA & ML TRAINING (Backend)
# ==========================================
# Real project-la neenga pd.read_csv('zomato_data.csv') use pannuvenga.
# Inga test panrathuku oru chinna dummy dataset create panrom.
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

# AI Pipeline: Text to Numbers -> Random Forest Classifier
ai_model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', RandomForestClassifier(n_estimators=50, random_state=42))
])

# AI-ah train panrom
ai_model.fit(df['Ingredients'], df['Cuisine'])

# ==========================================
# 2. WEB APP DESIGN (Frontend)
# ==========================================
st.set_page_config(page_title="Smart Cuisine AI", page_icon="🍽️")

st.title("🍽️ Smart Cuisine & Restaurant Finder")
st.write("Veetla irukkura ingredients-ah type pannunga, namma AI athu entha Cuisine nu kandupudichu, athuketha Restaurants-ah suggest pannum!")

st.markdown("---")

# User Input
st.subheader("👨‍🍳 What's in your fridge?")
user_ingredients = st.text_input("Example: tomato, cheese, basil")

# Prediction Button
if st.button("Predict Cuisine & Find Restaurants"):
    if user_ingredients:
        # 1. Predict the Cuisine
        predicted_cuisine = ai_model.predict([user_ingredients])[0]
        st.success(f"### 🪄 AI Predicts: **{predicted_cuisine} Cuisine**")
        
        st.markdown("---")
        
        # 2. Recommend Restaurants based on predicted cuisine
        st.subheader(f"🏆 Top {predicted_cuisine} Restaurants for you:")
        
        # Filter dataframe by the predicted cuisine and sort by rating
        recommended_df = df[df['Cuisine'] == predicted_cuisine].sort_values(by='Rating', ascending=False)
        
        if not recommended_df.empty:
            # Display beautifully using Streamlit metrics and dataframes
            for index, row in recommended_df.iterrows():
                col1, col2, col3 = st.columns(3)
                col1.metric(label="Restaurant", value=row['Restaurant_Name'])
                col2.metric(label="Rating", value=f"{row['Rating']} ⭐")
                col3.metric(label="Cost for Two", value=f"₹{row['Cost_for_Two']}")
                st.write("") # small spacing
        else:
            st.info("No restaurants found for this cuisine in our current database.")
            
    else:
        st.warning("Please enter some ingredients first!")