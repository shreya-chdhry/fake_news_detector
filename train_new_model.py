import pandas as pd
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# NLTK Stopwords download
nltk.download('stopwords', quiet=True)

print("Reading True and Fake news files...")
# Dono files ko read karo (Ensure names exactly match your files, e.g., True.csv and Fake.csv)
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

# Labels add karo (0 = Reliable/True, 1 = Unreliable/Fake)
true_df['label'] = 0
fake_df['label'] = 1

# Dono ko ek single dataset mein mix karo aur shuffle karo
df = pd.concat([true_df, fake_df], ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)  # Shuffling the data

# Sirf 'text' aur 'label' column rakho
df = df[['text', 'label']].dropna()

# Text Preprocessing
port_stem = PorterStemmer()
stop_words = set(stopwords.words('english'))

def stemming(content):
    con = re.sub('[^a-zA-Z]', ' ', str(content)).lower().split()
    return ' '.join([port_stem.stem(word) for word in con if word not in stop_words])

print("Cleaning and processing text (This might take 5-10 minutes, grab a coffee!)...")
# Note: Isme time lagta hai kyunki 40,000+ news articles clean ho rahe hain
df['text'] = df['text'].apply(stemming)

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.20, random_state=42)

print("Training the Logistic Regression Model...")
vect = TfidfVectorizer()
x_train_vect = vect.fit_transform(x_train)

model = LogisticRegression()
model.fit(x_train_vect, y_train)

# Naye updated brain (model) ko save karo
pickle.dump(vect, open('vector.pkl', 'wb'))
pickle.dump(model, open('model.pkl', 'wb'))
print("🎉 Smart model generated successfully! You can now run your app.")
x_test_vect = vect.transform(x_test)
y_pred = model.predict(x_test_vect)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")