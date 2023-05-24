import pickle

"""
This script will save a trained model to a file.
"""

def save_model(model, filename):
    """
    This function takes a trained model and a filename as input,
    and saves the model to a file.
    """
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

# Example usage:

# Save the model to a file
# save_model(model, 'model.pkl')


save_model(model, 'model.pkl')
print('model_saving.py has completed successfully. Proceed to model_loading.py...')


