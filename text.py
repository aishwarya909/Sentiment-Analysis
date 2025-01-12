import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import nltk

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Define stopwords
stop_words = set(stopwords.words('english'))
new_stopwords = [
    "mario", "la", "blah", "saturday", "monday", "sunday", "morning", "evening", "friday", 
    "would", "shall", "could", "might"
]
stop_words.update(new_stopwords)
stop_words.discard("not")

# Function to remove special characters
def remove_special_character(content):
    return re.sub('\W+', ' ', content)

# Function to remove URLs
def remove_url(content):
    return re.sub(r'http\S+', '', content)

# Function to remove stopwords
def remove_stopwords(content):
    clean_data = []
    for word in content.split():
        if word.lower() not in stop_words and word.isalpha():
            clean_data.append(word.lower())
    return " ".join(clean_data)

# Function to expand contractions
def contraction_expansion(content):
    contractions = {
        "won't": "would not", "can't": "can not", "don't": "do not",
        "shouldn't": "should not", "needn't": "need not", "hasn't": "has not",
        "haven't": "have not", "weren't": "were not", "mightn't": "might not",
        "didn't": "did not", "n't": " not"
    }
    for contraction, expansion in contractions.items():
        content = content.replace(contraction, expansion)
    return content

# Complete data cleaning function
def data_cleaning(content):
    try:
        content = contraction_expansion(content)
        content = remove_special_character(content)
        content = remove_url(content)
        content = remove_stopwords(content)
        return content
    except Exception as e:
        print(f"Error cleaning content: {content}")
        raise e

# Custom transformer for data cleaning
class DataCleaning(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        try:
            print("Applying data cleaning...")
            return X.apply(data_cleaning)
        except Exception as e:
            print(f"Error in transform method: {e}")
            raise e