import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer2.pkl','rb'))
model = pickle.load(open('model2.pkl','rb'))

# st.title("Email/SMS Spam Classifier")

# input_sms = st.text_area("Enter the message")

# if st.button('Predict'):

#     # 1. preprocess
#     transformed_sms = transform_text(input_sms)
#     # 2. vectorize
#     vector_input = tfidf.transform([transformed_sms])
#     # 3. predict
#     result = model.predict(vector_input)[0]
#     # 4. Display
#     if result == 1:
#         st.header("Spam")
#     else:
#         st.header("Not Spam")
# Custom CSS for modern UI design
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 95%);
    padding: 1rem;
}
.stTextArea > label {
    color: white !important;
    font-size: 1.2rem;
    font-weight: bold;
}
.stButton > button {
    background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 0.75rem 1rem;
    font-size: 1.1rem;
    font-weight: bold;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    transition: all 0.3s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
}
.result-spam {
    background: linear-gradient(45deg, #FF6B6B, #FF8E8E);
    padding: 2rem;
    border-radius: 20px;
    text-align: center;
    color: white;
    font-size: 2rem;
    font-weight: bold;
    box-shadow: 0 8px 25px rgba(255,107,107,0.4);
}
.result-ham {
    background: linear-gradient(45deg, #4ECDC4, #44A08D);
    padding: 2rem;
    border-radius: 20px;
    text-align: center;
    color: white;
    font-size: 2rem;
    font-weight: bold;
    box-shadow: 0 8px 25px rgba(78,205,196,0.4);
}
.metric-card {
    background: rgba(255,255,255,0.1);
    padding: 1.5rem;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.2);
}
</style>
""", unsafe_allow_html=True)

# Page config
st.set_page_config(page_title="Spam Classifier", page_icon="📧", layout="wide")

# Header
st.markdown("""
<div style='text-align: center; color: white; margin-bottom: 1.5rem;'>
    <h2 style='font-size: 3.5rem; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>📧 Spam Detector</h2>
    <p style='font-size: 1.1rem; opacity: 0.9;'>AI-powered SMS Classification</p>
</div>
""", unsafe_allow_html=True)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class='metric-card'>
        <h3 style='color: white; margin-top: 0;'>Enter your message :</h3>
    """, unsafe_allow_html=True)
    
    input_sms = st.text_area(
        "", 
        height=150,
        placeholder="Paste your email or SMS here to check whether it's a Spam or non-Spam Message ....",
        help="Enter the text you want to classify as spam or not spam."
    )

with col2:
    st.markdown("""
    <div class='metric-card'>
        <h4 style='color: white;'>🔍 Powered by:</h4>
        <ul style='color: white; list-style: none; padding: 0;'>
            <li>✅ MultinomialNB Model</li>
            <li>✅ Scikit-learn</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Prediction
if st.button('🔍 **CLASSIFY MESSAGE** 🚀', use_container_width=True):
    if input_sms.strip():
        with st.spinner('Analyzing your message...'):
            # Preprocess (assuming transform_text is defined elsewhere)
            try:
                transformed_sms = transform_text(input_sms)
                vector_input = tfidf.transform([transformed_sms])
                result = model.predict(vector_input)[0]
                confidence = model.predict_proba(vector_input)[0].max() * 100
                
                st.markdown("""
                <div style='text-align: center; margin: 1.1rem 0;'>
                """, unsafe_allow_html=True)
                
                if result == 1:
                    st.markdown(f"""
                    <div class='result-spam'>
                        <h>🚨 SPAM DETECTED</h>
                        <p>Confidence: <strong>{confidence:.1f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='result-ham'>
                        <h2>✅ SAFE MESSAGE</h2>
                        <p>Confidence: <strong>{confidence:.1f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Example messages
                
                        
            except Exception as e:
                st.error(f"❌ Processing error: {str(e)}")
    else:
        st.warning("⚠️ Please enter a message to classify!")

# Footer
st.markdown("""
<div style='text-align: center; color: rgba(255,255,255,0.8); padding: 0.5rem; border-top: 0.3px solid rgba(255,255,255,0.1); margin-top: 1.2rem;'>
    <p>Built with ❤️ By MOHD YASEEN ANSARI . . . . .</p>
</div>
""", unsafe_allow_html=True)