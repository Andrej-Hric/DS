# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Function to get k-mers from a sequence
def getKmers(sequence, size=6):
    if isinstance(sequence, str):  # Check if the sequence is not NaN
        return [sequence[x:x+size].lower() for x in range(len(sequence) - size + 1)]
    else:
        return []

# Function to extract features
def extract_features(csv_file_path):
    df = pd.read_csv(csv_file_path)
    
    # Handle NaN values in the 'Sequence' column
    df['Sequence'] = df['Sequence'].fillna('')
    
    df['words'] = df.apply(lambda x: getKmers(x['Sequence']), axis=1)
    df = df.drop('Sequence', axis=1)
    texts = list(df['words'])
    for item in range(len(texts)):
        texts[item] = ' '.join(texts[item])
    y_data = df.iloc[:, 0].values
    cv = CountVectorizer(ngram_range=(4,4))
    X = cv.fit_transform(texts)
    X_train, X_test, y_train, y_test = train_test_split(X, y_data, test_size=0.20, random_state=42)
    return X_train, X_test, y_train, y_test

# Example usage
X_train, X_test, y_train, y_test = extract_features('cleaned_data.csv')
