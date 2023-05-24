import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

def vectorize_sequences(sequences):
    # This function should transform a list of sequences into a matrix of shape (num_sequences, num_unique_words)
    # Each row of the matrix should correspond to one sequence
    # Each column of the matrix should correspond to one unique word in the sequences
    # Each entry of the matrix should correspond to the count of a word in a sequence

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sequences)
    return X

def main():
    # Read data
    data = pd.read_csv('/Users/andrejhric_1/git_projects/DataScience/Data_Science_Module_Project_predictor_of_cellular_compartment/predictor/cleaned_data.csv')

    # Get sequences (features) and labels
    X = data['Sequence'].values
    y = data['Subcellular_location'].values

    # Vectorize sequences
    X = vectorize_sequences(X)

    # Apply SMOTE for resampling
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    # Define hyperparameters for tuning
    params = {
        'C': [0.1, 1, 10, 100]
    }

    # Define model (Logistic Regression with balanced class weight)
    model = LogisticRegression(class_weight='balanced')

    # Define grid search
    clf = GridSearchCV(model, params, cv=5)

    # Perform hyperparameter tuning
    clf.fit(X_res, y_res)

    # Print the best parameter 'C'
    print(f"Best parameter C: {clf.best_params_['C']}")

# Run the main function
if __name__ == "__main__":
    main()
