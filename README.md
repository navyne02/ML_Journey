## Summary

This initial commit introduces the fundamental concepts of Machine Learning (ML) and sets up the local development environment. The goal is to shift from traditional programming mindsets to data-driven learning models.

## Changes

---

**1. The Paradigm Shift ( Traditional vs. ML )**

* **Traditional Programming:** `Input (Data) + Logic (Rules) ---> Output`
* **Machine Learning:** `Input (Data) + Expected Answers ---> Model (Rules)`

**2. ML Classifications ( Types of Models )**

* **Supervised Learning:** Learning with labels (e.g., Email Spam Detection).
* **Unsupervised Learning:** Finding hidden patterns in unlabeled data (e.g., YouTube Recommendations).
* **Reinforcement Learning:** Learning by trial and error using Rewards/Punishments (e.g., Game AI).

**3. Environment Setup ( local_machine )**

* Installed Python from `python.org`
* Installed Jupyter Notebook via `pip install jupyterlab`
* Installed data manipulation libraries `numpy` and `pandas`

---

**4. Day 2: Data Tables ( Pandas )**

* Learned to create a **DataFrame** (Table) using the `pandas` library.
* Organized raw data into rows and columns for better visualization.
* Handled file extension errors and Git merge conflicts.

**5. Day 3: Numerical Python ( NumPy )**

* Introduced **NumPy**, the foundation of math in Machine Learning.
* Learned that **Arrays** are much faster than normal Python lists.
* **Vectorization:** Performed math operations on entire lists without using loops.

---

**6. Day 4: Data Cleaning ( Handling Missing Data )**

* Learned that real-world data is often messy and contains missing values known as **NaN** (Not a Number).
* Used Pandas `fillna()` function to clean the dataset before feeding it to an ML model.
* Practiced two data imputation techniques:
  1. Filling missing categorical/numerical data with a default value (e.g., 0).
  2. Filling missing numerical data with the calculated **Mean** (Average).

 ---

**7. Day 5: Data Visualization ( Matplotlib )**

* Learned that visualizing data makes it easier to spot patterns and trends.
* Used the `matplotlib` library to draw graphs in Python.
* Created a Line Chart with customized markers, colors, and labels to track progress.

  ---

**8. Day 6: My First ML Model ( Linear Regression )**

* Installed and introduced `scikit-learn` library.
* Understood how a machine learns patterns from existing data using **Linear Regression**.
* Successfully trained a model to predict marks based on study hours!

---

**9. Day 7: Classification Models ( Logistic Regression )**

* Shifted from predicting continuous numbers (Regression) to predicting categories (Classification).
* Learned how **Logistic Regression** uses an S-Curve (Sigmoid function) to classify data into 0 or 1 (Fail/Pass).
* Successfully trained a model to classify a student's exam result based on study hours!

---

**10. Day 8: Unsupervised Learning ( K-Means Clustering )**

* Explored **Unsupervised Learning**, where the model finds hidden patterns without labeled answers.
* Used the **K-Means algorithm** to group unlabeled customer data into distinct segments.
* Visualized the AI-generated clusters using a scatter plot.

---

**11. Day 9: Model Evaluation ( Train / Test Split )**

* Learned the importance of not testing a model on the same data it was trained on.
* Used `train_test_split` from Scikit-Learn to divide data into **80% Training** and **20% Testing** sets.
* Evaluated the model's performance on unseen data using the `accuracy_score` metric.

---

**12. Day 10: Non-Linear Classification ( Decision Trees )**

* Learned that not all data can be separated by a simple line.
* Explored **Decision Trees**, which make predictions by asking a series of Yes/No questions.
* Built a model that acts like a flowchart to classify student performance based on attendance and study hours.

  ---

**13. Day 11: Ensemble Learning ( Random Forest )**

* Discovered the power of **Ensemble Learning**, where multiple models work together.
* Used **Random Forest Classifier** to combine the predictions of several Decision Trees via voting.
* Learned that a Random Forest prevents "overfitting" and provides much more accurate and stable results than a single tree.

---

**14. Day 12: Distance-Based ML ( K-Nearest Neighbors )**

* Learned the **K-Nearest Neighbors (KNN)** algorithm, which classifies data based on its proximity to other data points.
* Understood how the value of 'K' determines the number of neighbors to "vote" on a new data point's category.
* Built a KNN model to classify player roles based on performance statistics.

---

**15. Day 13: Deep Learning Basics ( Neural Networks )**

* Took the first step into **Deep Learning** by understanding Artificial Neural Networks.
* Learned how input data passes through **Hidden Layers** (neurons) to find complex, non-linear patterns.
* Built a `MLPClassifier` to predict outcomes based on the interaction between multiple variables (like study and sleep balance).

---

**16. Day 14: Maximum Margin Classifier ( SVM )**

* Learned how **Support Vector Machines (SVM)** find the best possible line (hyperplane) to separate data.
* Understood the concept of maximizing the "margin" between classes to improve model accuracy.
* Built a basic Spam Detector using `sklearn.svm.SVC` to classify messages based on specific features.

  ---

**17. Day 15: Data Pre-processing ( Feature Scaling )**

* Discovered the importance of **Feature Scaling** before feeding data into distance-based algorithms like KNN and SVM.
* Used `MinMaxScaler` from Scikit-Learn to normalize data features with drastically different ranges (e.g., Age vs. Salary).
* Learned that scaling transforms all features into a uniform range (usually 0 to 1), preventing larger numbers from dominating the AI model's logic.

---

**18. Day 16: Model Persistence ( Saving and Loading )**

* Learned that training a model every time a script runs is inefficient.
* Used the `joblib` library to serialize (save) a trained Machine Learning model into a `.pkl` file.
* Successfully deserialized (loaded) the saved model to make future predictions without retraining.

  ---

**19. Day 17: Model Optimization ( Hyperparameter Tuning )**

* Learned the difference between Model Parameters (learned from data) and **Hyperparameters** (set by the user).
* Understood the dangers of **Overfitting** (memorizing data) and **Underfitting** (failing to learn).
* Used `GridSearchCV` to automatically test multiple combinations of hyperparameters and find the most optimal settings for a model.

---

**20. Day 18: Streamlining Workflows ( ML Pipelines )**

* Learned how to automate the machine learning workflow using `sklearn.pipeline.Pipeline`.
* Combined data pre-processing (`MinMaxScaler`) and model training (`LogisticRegression`) into a single, cohesive object.
* Discovered how pipelines make code cleaner, prevent data leakage, and simplify making predictions on new data.

  ---

**21. Day 19: Natural Language Processing ( Sentiment Analysis )**

* Entered the world of **NLP (Natural Language Processing)** to teach the machine how to understand text.
* Used `CountVectorizer` to convert text data into numerical vectors (Bag of Words).
* Built a Sentiment Analysis model using the `MultinomialNB` (Naive Bayes) algorithm within a Pipeline to classify movie reviews as Positive or Negative.

  ---

**22. Day 20: Model Deployment ( Streamlit Web App )**

* Reached the final milestone: Deploying a Machine Learning model!
* Used the `streamlit` library to convert a Python ML script into a fully functional, interactive Web Application.
* Created a user interface with text inputs and buttons, allowing non-technical users to interact with the NLP Sentiment Analysis model easily.

  ---
  
**23. Bonus Project (Day 21): Smart Cuisine Predictor & Restaurant Finder 🍕**

* Built an end-to-end Machine Learning web application using **Streamlit**.
* Integrated **Natural Language Processing (`CountVectorizer`)** with a **`RandomForestClassifier`** inside an ML Pipeline to predict cuisines based on raw ingredients.
* Implemented a recommendation system to filter, rank, and suggest top-rated restaurants based on the AI's cuisine prediction.
* Transitioned from learning individual algorithms to building a complete, user-facing AI product suitable for portfolio display!

---

**24. Day 22: Content-Based Filtering ( Recommendation Systems ) 🎬**

* Built a simple Recommendation Engine inspired by platforms like Netflix and Spotify.
* Used **Natural Language Processing (`CountVectorizer`)** to convert movie tags and genres into numerical vectors.
* Applied **Cosine Similarity** (`sklearn.metrics.pairwise.cosine_similarity`) to mathematically calculate the distance/similarity between different movies.
* Created a function that takes a watched movie as input and outputs the top recommended movies based on feature overlap.

---

**25. Day 23: Intro to Computer Vision ( Image Classification ) 👁️**

* Took the first step into **Computer Vision** by understanding how computers process images as numerical arrays (pixels).
* Used the built-in `digits` dataset from Scikit-Learn to train a model on handwritten numbers.
* Applied the **Support Vector Machine (SVM)** algorithm to classify the 8x8 pixel images.
* Visualized the AI's prediction alongside the actual pixelated image using `matplotlib`.

---

**26. Day 24: Unsupervised Learning ( K-Means Clustering ) 🧩**

* Transitioned from Supervised to **Unsupervised Learning**, where data has no labels or answers.
* Learned the **K-Means Clustering** algorithm to automatically group data based on similarities (distance).
* Used `sklearn.cluster.KMeans` to segment a generated dataset into distinct customer groups.
* Visualized the clusters and their calculated centroids using `matplotlib`.

---

**27. Day 25: Dimensionality Reduction ( PCA ) 📉**

* Explored **Principal Component Analysis (PCA)**, an Unsupervised Learning technique used to compress data while retaining its core information.
* Used `sklearn.decomposition.PCA` to reduce the high-dimensional `digits` dataset (64 features) down to just 2 principal components.
* Successfully visualized a 64-dimensional dataset on a 2D scatter plot, observing how similar digits naturally cluster together in the reduced space.

---

**28. Day 26: Outlier Detection ( Anomaly Detection ) 🕵️‍♂️**

* Explored the critical field of **Anomaly Detection**, widely used in cyber-security and fraud detection.
* Used the **`IsolationForest`** algorithm to identify rare, suspicious data points (outliers) within a dataset.
* Built a simulated Credit Card Fraud Detection model that automatically separates normal transactions from anomalous ones without needing pre-labeled data.

---

**29. Day 27: Advanced NLP ( Fake News Detection ) 📰**

* Upgraded from simple Bag of Words to **TF-IDF (Term Frequency - Inverse Document Frequency)**.
* Understood how to penalize common stop words and assign higher weights to rare, significant words in a corpus.
* Built a text classification pipeline using `TfidfVectorizer` and `LogisticRegression` to detect sensationalized fake news headlines.

---

**30. Day 28: Sequential Ensemble Learning ( Gradient Boosting ) 👑**

* Learned the difference between parallel learning (Random Forest) and sequential learning (**Gradient Boosting**).
* Understood how Gradient Boosting builds trees one by one, where each new tree focuses on correcting the errors (residuals) of the previous ones.
* Implemented `GradientBoostingClassifier` to predict Customer Churn based on subscription duration, bill amount, and support tickets.
