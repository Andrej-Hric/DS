from sklearn import metrics

"""
This script will evaluate the performance of a model by calculating various metrics.
"""

def evaluate_model(y_true, y_pred):
    """
    This function takes the true and predicted labels as input,
    and prints various performance metrics.
    """
    # Calculate and print accuracy
    accuracy = metrics.accuracy_score(y_true, y_pred)
    print('Accuracy: ')
    print(accuracy)
    
    # Calculate and print precision, recall, and F1 score
    precision = metrics.precision_score(y_true, y_pred, average='micro')
    print('Precision: ')
    print(precision)

    recall = metrics.recall_score(y_true, y_pred, average='micro')
    print('Recall: ')
    print(recall)

    f1 = metrics.f1_score(y_true, y_pred, average='micro')
    print('F1: ')
    print(f1)

    # Calculate and print confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    print('Confusion Matrix:')
    print(cm)

# Example usage:

# y_true = true labels
# y_pred = predicted labels by your model

# evaluate_model(y_true, y_pred)


y_pred = model.predict(X_test)
evaluate_model(y_test, y_pred)
print('model_evaluation.py has completed successfully. All scripts have run.')


