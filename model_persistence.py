# Filename: model_persistence.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from joblib import dump, load

# create a simple classification dataset
X, y = make_classification(n_samples=100, n_features=4, random_state=0)

# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create and train a model
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, y_train)

# save the trained model to a file
dump(clf, 'model.joblib') 

# later... load the model from the file
clf_loaded = load('model.joblib') 

# make predictions using the loaded model
y_pred = clf_loaded.predict(X_test)
