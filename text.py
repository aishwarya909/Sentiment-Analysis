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
# Function to lemmatize text
def lemmatize_text(content):
    try:
        wordnetlemma = WordNetLemmatizer()
        return " ".join([wordnetlemma.lemmatize(word) for word in word_tokenize(content)])
    except Exception as e:
        print(f"Error during lemmatization: {content}")
        raise e
    
# Load dataset
data = pd.read_csv('eng_dataset.csv')

# Handle missing or null values in the 'content' column
data['content'] = data['content'].fillna('')
data['content'] = data['content'].astype(str)

# Encode the 'sentiment' column
data['sentiment'] = data['sentiment'].astype('category').cat.codes

# Apply custom cleaning and lemmatization
data_cleaner = DataCleaning()
data['content'] = data_cleaner.fit_transform(data['content'])
data['content'] = data['content'].apply(lemmatize_text)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
X = vectorizer.fit_transform(data['content'])
y = data['sentiment']

# Handle imbalanced data using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Define multiple models
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

# Train and evaluate models
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = None

    print(f"Model: {name}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    if y_proba is not None:
        try:
            if len(np.unique(y_test)) > 2:
                print(f"ROC-AUC: {roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr'):.4f}")
            else:
                print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
        except Exception as e:
            print(f"Error calculating ROC-AUC: {e}")

    print(classification_report(y_test, y_pred))
    print("-" * 50)

# Hyperparameter tuning for Random Forest
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10]
}

rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
rf_grid_search.fit(X_train, y_train)
print("Best Random Forest Parameters:", rf_grid_search.best_params_)

# Cross-validation
cv_scores = cross_val_score(RandomForestClassifier(**rf_grid_search.best_params_, random_state=42), X_resampled, y_resampled, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy: {np.mean(cv_scores):.4f}")
