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
