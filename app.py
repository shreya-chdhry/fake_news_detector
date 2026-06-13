import streamlit as st
import pickle
import re
import nltk
import pandas as pd
import csv
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# NLTK setup
nltk.download('stopwords', quiet=True)
port_stem = PorterStemmer()

# Load models
vector_form = pickle.load(open('vector.pkl', 'rb'))
load_model = pickle.load(open('model.pkl', 'rb'))

def stemming(content):
    con = re.sub('[^a-zA-Z]', ' ', content).lower().split()
    con = [port_stem.stem(word) for word in con if word not in stopwords.words('english')]
    return ' '.join(con)

def fake_news(news):
    news = stemming(news)
    input_data = [news]
    vector_form1 = vector_form.transform(input_data)
    prediction = load_model.predict(vector_form1)
    probability = load_model.predict_proba(vector_form1)
    return prediction, probability

# App UI
st.set_page_config(page_title="Fake News Detector", page_icon="🔍")
st.title('🔍 Fake News Classification App')
sentence = st.text_area("Enter your news content here:", "", height=200)

if st.button("Predict"):
    if sentence.strip() == "":
        st.error("Please enter some text to analyze!")
    else:
        prediction_class, probability = fake_news(sentence)
        prob_reliable = probability[0][0]
        prob_unreliable = probability[0][1]
        
        res_text = 'Reliable' if prediction_class[0] == 0 else 'Unreliable'
        
        st.markdown("---")
        if prediction_class[0] == 0:
            st.success(f'### Result: {res_text}')
        else:
            st.warning(f'### Result: {res_text}')

        # Save to session
        st.session_state['pred'] = res_text
        st.session_state['prob_r'] = prob_reliable
        st.session_state['prob_u'] = prob_unreliable
        st.session_state['sentence'] = sentence

# Display Results and Feedback
if 'pred' in st.session_state:
    st.subheader("Confidence Level")
    chart_data = pd.DataFrame({
        'Category': ['Reliable', 'Unreliable'], 
        'Confidence': [st.session_state['prob_r'], st.session_state['prob_u']]
    })
    st.bar_chart(chart_data.set_index('Category'))
    st.write(f"Model Confidence: **{max(st.session_state['prob_r'], st.session_state['prob_u'])*100:.2f}%**")

    st.markdown("---")
    st.subheader("Was this prediction correct?")
    feedback_choice = st.radio("Give feedback:", ("Yes, accurate", "No, incorrect"), key="fb")
    
    if st.button("Submit Feedback"):
        file_path = os.path.join(os.getcwd(), 'feedback.csv')
        file_exists = os.path.isfile(file_path)
        
        with open(file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['News', 'Prediction', 'Feedback'])
            writer.writerow([st.session_state['sentence'], st.session_state['pred'], feedback_choice])
        
        st.success("Feedback saved! Admins will use this to update the model later.")