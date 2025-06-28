
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load datasets
train_df = pd.read_csv('train.txt', sep=';', names=['text', 'emotion'])
val_df = pd.read_csv('val.txt', sep=';', names=['text', 'emotion'])
test_df = pd.read_csv('test.txt', sep=';', names=['text', 'emotion'])

# Combine datasets
full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# Text cleaning function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

full_df['clean_text'] = full_df['text'].apply(clean_text)

# Vectorize with TF-IDF bi-grams
vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1,2), min_df=5, max_df=0.7)
X = vectorizer.fit_transform(full_df['clean_text']).toarray()
y = full_df['emotion']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest with hyperparameter tuning
param_grid = {
    'n_estimators': [300, 400],
    'max_depth': [None, 20, 30],
    'min_samples_split': [2, 5]
}

grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

# Best model
best_model = grid.best_estimator_
print("Best parameters:", grid.best_params_)

# Cross-validation score
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
print("Cross-validation scores:", cv_scores)
print("Average CV score:", np.mean(cv_scores))

# Predict and evaluate
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Save model and vectorizer
joblib.dump(best_model, 'emotion_rf_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print(" Model and vectorizer saved.")
