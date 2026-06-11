# Fake News Classifier

A Machine Learning web application that classifies news articles as Reliable or Unreliable using NLP and a Streamlit frontend.

## Tech Stack
* **Language:** Python
* **ML & NLP:** Scikit-learn (Logistic Regression, TF-IDF), NLTK (Stemming, Stopwords)
* **Libraries:** Pandas, NumPy, Streamlit

## How to Run Locally

**1. Clone and open folder**
`git clone <your-repo-link>`
`cd fake_news_detection`

**2. Install requirements**
`pip install -r requirements.txt`

**3. Run the app**
`streamlit run app.py`

## Technical Limitations
* **Model Used:** Logistic Regression with TF-IDF Vectorizer.
* **Out-of-Distribution (OOD):** The model is highly accurate on its training domain (Political/World News). However, because TF-IDF relies on word frequency rather than semantic context, universally true statements (e.g., science facts) may be flagged as "Unreliable" if their vocabulary was absent from the training dataset.