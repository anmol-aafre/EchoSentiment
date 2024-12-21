import streamlit as st
import pickle
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import MinMaxScaler
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))

# Load models (make sure paths are correct)
model_path = 'D:/Data Science/project/'
try:
    model = pickle.load(open(model_path + 'model_xgb.pkl', 'rb'))
    cv = pickle.load(open(model_path + 'CountVectorizer.pkl', 'rb'))
    scaler = pickle.load(open(model_path + 'scalar.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"Error loading model files: {e}. Please check the file paths.")
    st.stop()  # Stop execution if files are not found

# Preprocessing function
stemmer = PorterStemmer()

def preprocess_review(review):
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower().split()
    review = [stemmer.stem(word) for word in review if word not in STOPWORDS]
    return ' '.join(review)

# Streamlit app
st.set_page_config(page_title="EchoSentiment", page_icon="😊")  # Set title and icon

# Use Markdown for larger title
st.markdown("<h1 style='text-align: center; font-size: 5em;'>EchoSentiment</h1>", unsafe_allow_html=True)

# Centered text and other styling
st.markdown("<p style='text-align: center;font-size: 2em;'>Understand the emotions behind the words 😊</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;font-size: 1em;'>Text sentiment prediction is a powerful tool that can help you to understand the emotions and opinions expressed in your text data. This information can be used to improve your business in a number of ways.</p>", unsafe_allow_html=True)

# Apply custom CSS to increase label font size
st.markdown("""
    <style>
        .big-font {
            font-size: 30px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Use the custom class for the label
st.markdown('<p class="big-font">Enter a review:</p>', unsafe_allow_html=True)

# Increased height for text area
review_input = st.text_area("", placeholder="Type your review here...", height=100)  # Empty label here as it's styled separately

if st.button("Predict Feedback"):
    if not review_input.strip():  # More Pythonic way to check for empty string
        st.warning("Please enter a review to predict!")
    else:
        preprocessed_review = preprocess_review(review_input)
        transformed_review = cv.transform([preprocessed_review]).toarray()
        scaled_review = scaler.transform(transformed_review)
        prediction = model.predict(scaled_review)
        feedback = "Positive 😊" if prediction[0] == 1 else "Negative 😞"
        st.success(f"The predicted feedback is: **{feedback}**")

# Footer with improved styling and black horizontal line
st.markdown("<hr style='border: 0.5px solid gray;'>", unsafe_allow_html=True)  # Horizontal line
st.markdown("<p style='text-align: center; font-size: 1em; color: white;'>Developed with ❤️ by Anmol Aafre</p>", unsafe_allow_html=True)












