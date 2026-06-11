import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ye line bohot important hai warna app crash hogi
nltk.download('stopwords', quiet=True)

port_stem = PorterStemmer()

# Pre-trained models load karna
vector_form = pickle.load(open('vector.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))

def stemming(content):
    con = re.sub('[^a-zA-Z]', ' ', content)
    con = con.lower()
    con = con.split()
    con = [port_stem.stem(word) for word in con if word not in stopwords.words('english')]
    return ' '.join(con)

def fake_news(news):
    news = stemming(news)
    input_data = [news]
    vector_form1 = vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    return prediction

if __name__ == '__main__':
    st.title('Fake News Classification App')
    st.subheader("Input the News content below")
    
    sentence = st.text_area("Enter your news content here", "", height=200)
    predict_btt = st.button("Predict")
    
    if predict_btt:
        prediction_class = fake_news(sentence)
        if prediction_class[0] == 0:
            st.success('Reliable')
        else:
            st.warning('Unreliable')