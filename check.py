import pickle
# Quick retrain script - save as retrain.py
import pickle
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Download nltk resources if you haven't already
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Text preprocessing function same as in app.py
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    text = y[:]
    y.clear()
    y = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    text = y[:]
    y.clear()
    y = [ps.stem(i) for i in text]
    return " ".join(y)

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin1')  # fix encoding

# Inspect columns and adjust depending on your actual csv structure
print(df.columns)
# Usually for SMS Spam dataset: 'v1' is label, 'v2' is message text, adjust if needed
X = df['v2'].apply(transform_text)
y = df['v1'].map({'ham':0, 'spam':1})  # encoding labels to 0 and 1

print(X,y)
exit()

# Vectorize
tfidf = TfidfVectorizer(max_features=3000)
X_vec = tfidf.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vec, y)

# Save both vectorizer and model pickle files for your app.py to use
pickle.dump(tfidf, open('vectorizer2.pkl', 'wb'))
pickle.dump(model, open('model2.pkl', 'wb'))

print(" Model and vectorizer saved successfully!")
