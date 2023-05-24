from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import pandas as pd
import numpy as np
import pickle

"""
This script will build a model using RandomForestClassifier.
"""

def load_data(filename, target_column='Subcellular location [CC]'):
    # Load the data
    df = pd.read_csv(filename)
    print(f"Columns in DataFrame: {df.columns}")
    
    # Clean the target column (Remove rows with nan values)
    df = df.dropna(subset=[target_column])
    
    # Vectorize the 'Sequence' feature
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(4, 4))
    X = vectorizer.fit_transform(df['Sequence'])

    # Convert the target column to string type
    y = df[target_column].astype(str)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def build_and_train_model(X_train, y_train, best_params):
    """
    This function takes the train set and the best parameters as input,
    builds a model using RandomForestClassifier with the best parameters,
    trains the model on the train set, and returns the trained model.
    """
    # Instantiate the classifier with the best parameters
    rf = RandomForestClassifier(**best_params)
    
    # Train the model
    rf.fit(X_train, y_train)
    
    return rf

# Load the data
X_train, X_test, y_train, y_test = load_data('cleaned_data.csv')

# Define the best parameters. These are usually output from hyperparameter tuning.
best_params = {'n_estimators': 100, 'max_depth': None, 'random_state': 42}  # Replace these values with your actual best parameters

# Build and train the model
model = build_and_train_model(X_train, y_train, best_params)

# Save the model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
