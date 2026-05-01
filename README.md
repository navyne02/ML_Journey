<div align="center">

```
 ___  ___  __       _  ___  _   _ ___ _  _ _________   __
|   \|   \/  |     | |/ _ \| | | | _ \ \| | __\ \ /  / /
| |) | |) | |_  _  | | (_) | |_| |   / .  | _|  \ \ / /
|___/|___/|___||_|_| |\___/ \___/|_|_\_|\_|___|  |_|/_/
             |___|___|
```

# ⚡ 30-Day Advanced Machine Learning & AI Challenge

> *"From zero to deploying AI — one commit, one concept, one day at a time."*

[![Days Completed](https://img.shields.io/badge/Days%20Completed-30%2F30-brightgreen?style=for-the-badge&logo=checkmarx)](.)
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](./LICENSE)
[![Made With Python](https://img.shields.io/badge/Made%20With-Python%203.x-yellow?style=for-the-badge&logo=python)](.)
[![Status](https://img.shields.io/badge/Status-Completed%20🎉-blueviolet?style=for-the-badge)](.)

</div>

---

## 🧭 The Paradigm Shift

Before writing a single line of ML code, I had to rewire my thinking:

| Traditional Programming | Machine Learning |
|:---:|:---:|
| `Data + Rules → Output` | `Data + Answers → Rules (Model)` |
| You define the logic | The machine discovers the logic |
| Rigid, hand-crafted | Adaptive, data-driven |

This shift — from **telling** a computer what to do, to **showing** it — is the heart of everything in this challenge.

---

## 🗺️ The 30-Day Roadmap

```
FOUNDATIONS       →    MODELS        →    ADVANCED AI     →    DEPLOYMENT
──────────────         ──────────         ───────────────       ──────────
Day 1  Setup           Day 6  LinReg      Day 19 NLP            Day 20 Streamlit
Day 2  Pandas          Day 7  LogReg      Day 22 RecSys         Day 21 Cuisine AI
Day 3  NumPy           Day 8  KMeans      Day 23 CV             Day 29 GPT-2
Day 4  Cleaning        Day 9  Eval        Day 26 Anomaly        Day 30 AI Chatbot
Day 5  Matplotlib      Day 10 Trees       Day 27 FakeNews
                       Day 11 Forest      Day 28 GradBoost
                       Day 12 KNN         Day 15 Scaling
                       Day 13 NeuralNet   Day 17 Tuning
                       Day 14 SVM         Day 18 Pipelines
                       Day 25 PCA         Day 16 Persistence
```

---

## 📅 Day-by-Day Breakdown

### 🏗️ Phase 1 — Foundations (Days 1–5)

<details>
<summary><b>Day 1 · The Big Picture & Environment Setup</b></summary>

- Understood the three pillars of ML: **Supervised**, **Unsupervised**, and **Reinforcement Learning**
- Set up a local Python development environment with **JupyterLab**
- Installed core libraries: `numpy`, `pandas`

```
Supervised    → Learning with labels         (Email Spam Detection)
Unsupervised  → Finding hidden patterns      (YouTube Recommendations)
Reinforcement → Learning via rewards         (Game AI)
```
</details>

<details>
<summary><b>Day 2 · Data Tables with Pandas</b></summary>

- Created **DataFrames** — the fundamental data structure of ML workflows
- Organized raw data into rows and columns for analysis
- Handled real-world file extension errors and Git merge conflicts
</details>

<details>
<summary><b>Day 3 · Numerical Python with NumPy</b></summary>

- Learned why **Arrays** outperform Python lists for ML
- Mastered **Vectorization**: applying math to entire arrays without loops
- Laid the mathematical foundation for every algorithm to come
</details>

<details>
<summary><b>Day 4 · Data Cleaning — Handling Missing Values</b></summary>

- Real-world data is messy. Learned to handle **NaN (Not a Number)** values
- Used `fillna()` for two data imputation strategies:
  - Fill with a **default value** (categorical / numerical)
  - Fill with the **Mean** (average) of the column
</details>

<details>
<summary><b>Day 5 · Data Visualization with Matplotlib</b></summary>

- Built line charts with custom markers, colors, and labels
- Learned that a good visualization reveals patterns no table can
</details>

---

### 🤖 Phase 2 — Core ML Algorithms (Days 6–18)

<details>
<summary><b>Day 6 · My First ML Model — Linear Regression</b></summary>

- Introduced `scikit-learn`, the Swiss army knife of ML
- **Linear Regression**: finds the best-fit line to predict continuous values
- 🎯 **Project**: Predicted exam marks based on study hours
</details>

<details>
<summary><b>Day 7 · Classification — Logistic Regression</b></summary>

- Shifted from predicting numbers to predicting **categories**
- **Sigmoid function (S-Curve)**: maps any number to a probability (0 to 1)
- 🎯 **Project**: Classified student exam results (Pass / Fail)
</details>

<details>
<summary><b>Day 8 · Unsupervised Learning — K-Means Clustering</b></summary>

- No labels. No answers. The model finds its own patterns
- **K-Means**: groups data points into K clusters by minimizing distance
- 🎯 **Project**: Segmented unlabeled customer data and visualized the clusters
</details>

<details>
<summary><b>Day 9 · Model Evaluation — Train / Test Split</b></summary>

- A model tested on its own training data is a cheater
- Split data: **80% Training | 20% Testing**
- Evaluated using `accuracy_score` on truly unseen data
</details>

<details>
<summary><b>Day 10 · Non-Linear Classification — Decision Trees</b></summary>

- Not all decisions are a straight line
- **Decision Trees**: ask a series of Yes/No questions to classify data
- 🎯 **Project**: Classified student performance via attendance + study hours
</details>

<details>
<summary><b>Day 11 · Ensemble Learning — Random Forest</b></summary>

- *Why trust one tree when you can have an entire forest?*
- **Random Forest**: combines predictions from many Decision Trees via majority vote
- Naturally prevents **overfitting** — more stable and accurate than a single tree
</details>

<details>
<summary><b>Day 12 · Distance-Based ML — K-Nearest Neighbors (KNN)</b></summary>

- "You are the average of your K closest neighbors"
- **KNN**: classifies a point based on the majority label of its K neighbors
- 🎯 **Project**: Classified player roles based on performance statistics
</details>

<details>
<summary><b>Day 13 · Deep Learning Basics — Neural Networks</b></summary>

- First steps into the brain of AI
- **MLP (Multi-Layer Perceptron)**: input → hidden layers → output
- 🎯 **Project**: Predicted outcomes based on multi-variable interaction (study + sleep balance)
</details>

<details>
<summary><b>Day 14 · Maximum Margin Classifier — SVM</b></summary>

- **SVM**: finds the hyperplane with the *maximum margin* between classes
- Extremely effective in high-dimensional spaces
- 🎯 **Project**: Built a basic Spam Detector using `sklearn.svm.SVC`
</details>

<details>
<summary><b>Day 15 · Data Pre-processing — Feature Scaling</b></summary>

- Without scaling, features with large values dominate the model
- `MinMaxScaler`: squishes all features into a uniform **[0, 1]** range
- Essential for distance-based algorithms: KNN, SVM
</details>

<details>
<summary><b>Day 16 · Model Persistence — Save & Load with Joblib</b></summary>

- Retraining from scratch every time is inefficient
- `joblib.dump()` → serializes model to `.pkl` file
- `joblib.load()` → restores the model for instant predictions
</details>

<details>
<summary><b>Day 17 · Model Optimization — Hyperparameter Tuning</b></summary>

- **Parameters**: learned by the model from data
- **Hyperparameters**: set by you, before training
- Used `GridSearchCV` to automatically find the optimal combination
- Addressed **Overfitting** (memorizes data) vs. **Underfitting** (fails to learn)
</details>

<details>
<summary><b>Day 18 · Streamlining Workflows — ML Pipelines</b></summary>

- Chained `MinMaxScaler` + `LogisticRegression` into a single `Pipeline` object
- Cleaner code. No data leakage. Simpler predictions on new data.
</details>

---

### 🧠 Phase 3 — Advanced AI & Specializations (Days 19–29)

<details>
<summary><b>Day 19 · Natural Language Processing — Sentiment Analysis</b></summary>

- Taught the machine to understand human text
- **Bag of Words** (`CountVectorizer`): converts text → numerical vectors
- **Naive Bayes** (`MultinomialNB`): probabilistic classification inside a Pipeline
- 🎯 **Project**: Classified movie reviews as Positive or Negative
</details>

<details>
<summary><b>Day 22 · Recommendation Systems — Content-Based Filtering</b></summary>

- Inspired by Netflix & Spotify recommendation engines
- `CountVectorizer` + **Cosine Similarity** to measure movie feature overlap
- 🎯 **Project**: Built a movie recommender that outputs top picks for any watched film
</details>

<details>
<summary><b>Day 23 · Computer Vision — Image Classification</b></summary>

- Computers see images as pixel arrays. Learned to work with that
- Used the `digits` dataset (8×8 pixel handwritten numbers)
- **SVM** trained to classify all 10 digit classes
- Visualized predictions alongside the actual pixel images
</details>

<details>
<summary><b>Day 24 · Unsupervised Learning — K-Means (Advanced)</b></summary>

- Deep-dived into clustering with generated datasets
- Visualized clusters and their **centroids** for business interpretation
</details>

<details>
<summary><b>Day 25 · Dimensionality Reduction — PCA</b></summary>

- **PCA**: compresses data while retaining maximum information
- Reduced the `digits` dataset from **64 features → 2 principal components**
- Visualized a 64-dimensional dataset on a 2D scatter plot 🤯
</details>

<details>
<summary><b>Day 26 · Anomaly Detection — Fraud Detection</b></summary>

- No labels. No supervision. Find the outliers automatically
- **IsolationForest**: isolates rare, anomalous data points efficiently
- 🎯 **Project**: Simulated credit card fraud detection system
</details>

<details>
<summary><b>Day 27 · Advanced NLP — Fake News Detection with TF-IDF</b></summary>

- Upgraded from Bag of Words → **TF-IDF**
- Penalizes common words, rewards rare & significant ones
- 🎯 **Project**: Text classifier to detect sensationalized fake news headlines
</details>

<details>
<summary><b>Day 28 · Sequential Ensemble — Gradient Boosting</b></summary>

- **Random Forest**: trains trees in parallel
- **Gradient Boosting**: trains trees sequentially, each correcting the last
- 🎯 **Project**: Predicted customer churn using `GradientBoostingClassifier`
</details>

<details>
<summary><b>Day 29 · Large Language Models — Transformers & Hugging Face</b></summary>

- Entered the era of Generative AI
- Used `transformers` library to load a pre-trained **GPT-2** model
- 🎯 **Project**: Text generation pipeline that auto-completes user prompts
</details>

---

### 🚀 Phase 4 — Deployment & Real Products (Days 20–21, 30)

<details>
<summary><b>Day 20 · Model Deployment — Streamlit Web App</b></summary>

- Converted a Python ML script into an interactive web application
- Text inputs, buttons, and live predictions — no web dev experience needed
- 🎯 **Project**: Deployed the NLP Sentiment Analysis model for real users
</details>

<details>
<summary><b>Day 21 · Bonus Project — Smart Cuisine Predictor & Restaurant Finder 🍕</b></summary>

- End-to-end ML product: NLP → RandomForest → Recommendation → Streamlit UI
- Users enter ingredients → AI predicts cuisine → Recommends top-rated restaurants
- 🎯 First portfolio-ready, user-facing AI application!
</details>

<details>
<summary><b>Day 30 · Grand Finale — Full-Stack AI Chatbot 🤖</b></summary>

- Combined **Streamlit** + **Hugging Face Transformers** into one app
- `st.chat_message` + `st.session_state` for persistent conversation history
- Local **GPT-2** model powers real-time text generation
- 🎉 **Challenge Complete!**
</details>

---

## 🛠️ Tech Stack

```python
tech_stack = {
    "Language"         : "Python 3.x",
    "Notebook"         : "JupyterLab",
    "Data"             : ["NumPy", "Pandas"],
    "Visualization"    : ["Matplotlib"],
    "ML Core"          : ["Scikit-Learn"],
    "Deep Learning"    : ["MLPClassifier (sklearn)"],
    "NLP"              : ["CountVectorizer", "TfidfVectorizer", "Naive Bayes"],
    "Generative AI"    : ["Hugging Face Transformers", "GPT-2"],
    "Deployment"       : ["Streamlit"],
    "Persistence"      : ["Joblib"],
}
```

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ml-journey.git
cd ml-journey

# Install all dependencies
pip install jupyterlab numpy pandas matplotlib scikit-learn \
            streamlit joblib transformers torch

# Launch Jupyter
jupyter lab
```

---

## 🏆 Milestones Unlocked

```
✅  Built & deployed 5+ real-world ML projects
✅  Mastered 10+ core ML algorithms from scratch
✅  Entered Computer Vision, NLP, and Generative AI
✅  Deployed a live Streamlit web application
✅  Built a full-stack AI Chatbot using GPT-2
✅  Completed the 30-Day ML & AI Challenge
```

---

## 📁 Project Structure

```
ml-journey/
│
├── 📓 notebooks/
│   ├── day_01_intro.ipynb
│   ├── day_02_pandas.ipynb
│   ├── ...
│   └── day_30_chatbot.ipynb
│
├── 🚀 apps/
│   ├── sentiment_app.py       # Day 20 — Streamlit Sentiment Analyzer
│   ├── cuisine_predictor.py   # Day 21 — Smart Cuisine Predictor
│   └── chatbot_app.py         # Day 30 — AI Chatbot
│
├── 💾 models/
│   └── *.pkl                  # Saved trained models (Joblib)
│
├── 📄 README.md
└── 📜 LICENSE
```

---

## 🙏 Acknowledgements

- [Scikit-Learn Documentation](https://scikit-learn.org/)
- [Hugging Face](https://huggingface.co/) for democratizing LLMs
- [Streamlit](https://streamlit.io/) for making ML deployment effortless
- Every `StackOverflow` answer that saved a debug session at 2am 🌙

---

<div align="center">

**Made with 🔥 curiosity, ☕ coffee, and 30 days of commitment.**

*If this journey inspired you — give it a ⭐ and start your own.*

</div>

---

## 📜 License

```
MIT License

Copyright (c) 2025 Navyne02

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
