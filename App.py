import streamlit as st
import pickle
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from wordcloud import WordCloud
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

# Load Dataset
dataset = pd.read_csv(r"C:\Users\divya\OneDrive\Documents\Sentiment analysis\Restaurant_Reviews.tsv", delimiter='\t', quoting=3)

# Check for Sentiment column
if 'Liked' in dataset.columns:
    sentiment_column = 'Liked'
else:
    st.error("Sentiment column not found. Please check your dataset.")
    sentiment_column = None

if sentiment_column:
    # Define classifiers
    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(),
        'Support Vector Classifier': SVC(),
        'AdaBoost': AdaBoostClassifier(),
        'Gradient Boosting': GradientBoostingClassifier()
    }

    # Preprocessing Function
    def preprocess_review(review):
        ps = PorterStemmer()
        review = re.sub('[^a-zA-Z]', ' ', review)
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
        return ' '.join(review)

    # Streamlit App
    st.title("Sentiment Analysis Web App")
    st.write("""
        Enter a review to predict whether it is positve or negative.
    """)

    # Sidebar for Model Selection
    model_name = st.sidebar.selectbox("Select Machine Learning Model", list(models.keys()))

    # Split the dataset for training and testing
    X = dataset['Review']
    y = dataset[sentiment_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    # Load the vectorizer
    with open(r'C:\Users\divya\OneDrive\Documents\Sentiment analysis\count_vectorizer.pkl', 'rb') as vectorizer_file:
        cv = pickle.load(vectorizer_file)

    # Load Model (based on user selection)
    classifier = models[model_name]
    classifier.fit(cv.transform(X_train).toarray(), y_train)

    # Input from the user
    user_input = st.text_area("Enter a review:")

    if st.button("Analyze Sentiment"):
        with st.spinner('Analyzing your review...'):
            # Preprocess and transform the input
            processed_review = preprocess_review(user_input)
            input_vectorized = cv.transform([processed_review]).toarray()

            # Predict sentiment
            prediction = classifier.predict(input_vectorized)

            # Display result
            if prediction[0] == 1:
                st.success("Positive Review! 🎉")
            else:
                st.error("Negative Review 😞")

            # Display model performance evaluation
            st.subheader(f"Model Evaluation for {model_name}:")
            y_pred = classifier.predict(cv.transform(X_test).toarray())  # Make predictions on test set
            cm = confusion_matrix(y_test, y_pred)  # Confusion Matrix
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))
            st.write(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")

            # Visual Insights: Confusion Matrix and Distribution Side by Side
            st.subheader("Confusion Matrix and Distribution of Reviews")
            col1, col2 = st.columns(2)  # Create two columns

            # Confusion Matrix Heatmap
            with col1:
                st.write("Confusion Matrix Heatmap")
                fig1, ax1 = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax1)
                ax1.set_xlabel("Predicted")
                ax1.set_ylabel("Actual")
                st.pyplot(fig1)

            # Distribution of Positive vs Negative Reviews
            with col2:
                st.write("Distribution of Positive vs Negative Reviews")
                positive_count = sum(y_test == 1)
                negative_count = len(y_test) - positive_count
                fig2, ax2 = plt.subplots()
                ax2.bar(["Positive", "Negative"], [positive_count, negative_count], color=["green", "red"])
                ax2.set_ylabel("Count")
                st.pyplot(fig2)

    # Visual Insights: Word Cloud
    st.subheader("Word Cloud Visualization of Reviews")

    # Generate Word Cloud for positive and negative reviews
    positive_reviews = [review for review, sentiment in zip(dataset['Review'], dataset[sentiment_column]) if sentiment == 1]
    negative_reviews = [review for review, sentiment in zip(dataset['Review'], dataset[sentiment_column]) if sentiment == 0]

    positive_words = ' '.join(positive_reviews)
    negative_words = ' '.join(negative_reviews)

    positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_words)
    negative_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_words)

    # Display the word clouds side by side
    col3, col4 = st.columns(2)  # Create two columns

    with col3:
        st.image(positive_wordcloud.to_array(), caption='Positive Reviews Word Cloud', use_container_width=True)

    with col4:
        st.image(negative_wordcloud.to_array(), caption='Negative Reviews Word Cloud', use_container_width=True)
