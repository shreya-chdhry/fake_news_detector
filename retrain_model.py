import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('stopwords', quiet=True)
port_stem = PorterStemmer()
stop_words = set(stopwords.words('english'))

def stemming(content):
    con = re.sub('[^a-zA-Z]', ' ', str(content)).lower().split()
    return ' '.join([port_stem.stem(word) for word in con if word not in stop_words])

true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")
true_df['label'] = 0
fake_df['label'] = 1
df = pd.concat([true_df, fake_df], ignore_index=True)

try:
    feedback_df = pd.read_csv("feedback.csv")
    incorrect_feedback = feedback_df[feedback_df['Feedback'] == 'No, incorrect']
    
    if not incorrect_feedback.empty:
        incorrect_feedback = incorrect_feedback.copy()
        incorrect_feedback['label'] = incorrect_feedback['Prediction'].apply(lambda x: 1 if x == 'Reliable' else 0)
        incorrect_feedback = incorrect_feedback[['News', 'label']].rename(columns={'News': 'text'})
        
        df = pd.concat([df, incorrect_feedback], ignore_index=True)
except FileNotFoundError:
    pass

df = df.dropna()
df['text'] = df['text'].apply(stemming)

vect = TfidfVectorizer()
X = vect.fit_transform(df['text'])
y = df['label']

model = LogisticRegression()
model.fit(X, y)

pickle.dump(vect, open('vector.pkl', 'wb'))
pickle.dump(model, open('model.pkl', 'wb'))