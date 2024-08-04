import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from io import BytesIO
import pdfplumber

nltk.download('punkt')
nltk.download('stopwords')

# Load models
clf = pickle.load(open('First_NLP_Model.pkl', 'rb'))
vector = pickle.load(open('vector.pkl', 'rb'))

def clean_resume(resume_text):
    # Initialize the PorterStemmer
    ps = PorterStemmer()
    # Get English stopwords
    all_stopwords = set(stopwords.words('english'))

    # Remove unwanted characters, convert to lowercase, and split into words
    review = re.sub(r'[^a-zA-Z0-9]', ' ', resume_text)
    review = review.lower()
    review = review.split()

    # Remove stopwords and apply stemming
    review = [ps.stem(word) for word in review if word not in all_stopwords]

    # Join the words back into a single string
    cleaned_text = ' '.join(review)

    return cleaned_text

def read_pdf(file):
    """Extract text from a PDF file using pdfplumber."""
    with pdfplumber.open(BytesIO(file)) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

def main():
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader('Upload Resume', type=['txt', 'pdf'])

    if uploaded_file is not None:
        try:
            if uploaded_file.type == "application/pdf":
                # Use pdfplumber to extract text from PDF
                resume_text = read_pdf(uploaded_file.read())
            else:
                # Assume it's a text file and decode it
                resume_bytes = uploaded_file.read()
                resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # Handle different encoding issues
            resume_text = resume_bytes.decode('latin-1')

        # Clean the resume text
        cleaned_resume = clean_resume(resume_text)

        # Transform the text using the vectorizer
        input_features = vector.transform([cleaned_resume])

        # Make a prediction
        prediction_id = clf.predict(input_features)[0]

        # Display the prediction
        # st.write(f"Prediction: {prediction_id}")
        # Map category ID to category name
        category_mapping = {
            15: "Java Developer",
            23: "Testing",
            8: "DevOps Engineer",
            20: "Python Developer",
            24: "Web Designing",
            12: "HR",
            13: "Hadoop",
            3: "Blockchain",
            10: "ETL Developer",
            18: "Operations Manager",
            6: "Data Science",
            22: "Sales",
            16: "Mechanical Engineer",
            1: "Arts",
            7: "Database",
            11: "Electrical Engineering",
            14: "Health and fitness",
            19: "PMO",
            4: "Business Analyst",
            9: "DotNet Developer",
            2: "Automation Testing",
            17: "Network Security Engineer",
            21: "SAP Developer",
            5: "Civil Engineer",
            0: "Advocate",
        }
        category_name = category_mapping.get(prediction_id, "Unknown")

        st.write("Predicted Category:", category_name)

if __name__ == "__main__":
    main()
