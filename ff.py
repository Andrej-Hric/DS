import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def load_data(filepath):
    df = pd.read_csv(filepath)

    # Option 1: Drop rows where 'Subcellular location [CC]' is NaN
    df = df.dropna(subset=['Subcellular location [CC]'])

    # Option 2: Fill NaN values in 'Subcellular location [CC]' with 'unknown'
    # df['Subcellular location [CC]'].fillna('unknown', inplace=True)

    X = df['Sequence']
    y = df['Subcellular location [CC]']

    return X, y

def vectorize_sequences(sequences):
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(4,4))
    X = vectorizer.fit_transform(sequences)
    return X

def tune_hyperparameters(X, y):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    clf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3)
    grid_search.fit(X, y)
    return grid_search.best_params_

def main():
    X, y = load_data('cleaned_data.csv')
    X = vectorize_sequences(X)
    best_params = tune_hyperparameters(X, y)
    print(best_params)

if __name__ == "__main__":
    main()
