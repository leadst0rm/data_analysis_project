import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(raw_text):

    # Ensure the input is a string and handle NaN values
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)

    # Lowercase the text
    raw_text = raw_text.lower()

    # Substitute all underscore symbols "_" for whitespaces " "
    cleaned_text = re.sub(r"_", " ", raw_text)
    # Remove punctuation and special characters from the text
    cleaned_text = re.sub(r'[^\w\s]', '', cleaned_text)
    # Remove digits from the text
    cleaned_text = re.sub(r'\d+', '', cleaned_text)
    # Replace multiple spaces with a single space
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    # Tokenize the text
    tokens = word_tokenize(cleaned_text)

    # Remove stopwords from the tokenized text
    stop_words = set(stopwords.words('english'))
    filtered_tokens = []
    for word in tokens:
        if word not in stop_words:
            filtered_tokens.append(word)

    # Lemmatize the tokens
    lemmatized_tokens = [WordNetLemmatizer().lemmatize(word) for word in filtered_tokens]

    # Combine tokens back into a single string
    return ' '.join(lemmatized_tokens)

# Load data and apply the preprocessing function
complaints = pd.read_csv('complaints.csv')
complaints['preprocessed_complaints'] = complaints['Request Details'].apply(preprocess_text)

# Create a Bag of Words model
BoW_vectorizer = CountVectorizer(stop_words='english')
# Convert to a sparse matrix
BoW_matrix = BoW_vectorizer.fit_transform(complaints['preprocessed_complaints'])
# Sum up the BoW scores for each word across all complaints
BoW_sum = BoW_matrix.sum(axis=0).A1
# Create a Pandas Series with words as the index and their total occurrences as values
BoW_sum_series = pd.Series(BoW_sum, index=BoW_vectorizer.get_feature_names_out())

# Create a TF-IDF model
TF_IDF_vectorizer = TfidfVectorizer(stop_words='english')
# Convert to a sparse matrix
TF_IDF_matrix = TF_IDF_vectorizer.fit_transform(complaints['preprocessed_complaints'])
# Sum up the TF-IDF scores for each word across all complaints
TF_IDF_sum = TF_IDF_matrix.sum(axis=0).A1
# Create a Pandas Series with words as the index and their total TF-IDF scores as values
TF_IDF_sum_series = pd.Series(TF_IDF_sum, index=TF_IDF_vectorizer.get_feature_names_out())

# Comparing BoW and TF-IDF
print("Top 10 words for BoW:")
print(BoW_sum_series.sort_values(ascending=False).head(10))
print("\nTop 10 words for TF-IDF:")
print(TF_IDF_sum_series.sort_values(ascending=False).head(10))

# Prints top words for each topic for both LDA and NMF models
def print_topics(model, tfidf_vectorizer, top_n_words=10):

    # Get the words from the TF-IDF vectorizer
    words = tfidf_vectorizer.get_feature_names_out()

    # Loop through each topic and print its top words
    for topic_index, topic in enumerate(model.components_):

        # Get the indices of the top words for the topic
        top_word_indices = topic.argsort()[:-top_n_words - 1:-1]

        # Get the corresponding words for the top indices
        top_words = [words[i] for i in top_word_indices]

        # Print the topic with the top words
        print(f"Topic {topic_index + 1}: {' '.join(top_words)}")

# Define the number of topics
num_of_topics = 5

# Initialize the LDA model and fit to the TF-IDF matrix
LDA_model = LatentDirichletAllocation(n_components=num_of_topics, random_state=42) # Setting random_state to the same value ensures everyone can replicate the same results
LDA_model.fit(TF_IDF_matrix)

# Initialize the NMF model and fit to the TF-IDF matrix
NMF_model = NMF(n_components=num_of_topics, random_state=42) # Setting random_state to the same value ensures everyone can replicate the same results
NMF_model.fit(TF_IDF_matrix)

# Display the top words for both LDA and NMF topics
print("\nLDA Topics:")
print_topics(LDA_model, TF_IDF_vectorizer)
print("\nNMF Topics:")
print_topics(NMF_model, TF_IDF_vectorizer)