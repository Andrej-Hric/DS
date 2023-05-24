# Import the necessary libraries
import pandas as pd
import re
import os

# Function to clean a protein sequence
def clean_sequence(seq):
    if type(seq) == str:
        pattern = re.compile(r'[^ACDEFGHIKLMNPQRSTVWYUO]')
        cleaned_seq = pattern.sub('', seq.upper())
    else:
        cleaned_seq = ''
    return cleaned_seq

# Function to load and clean the data
def load_and_clean_data(tsv_file_path):
    df = pd.read_csv(tsv_file_path, sep='\t')
    df['Sequence'] = df['Sequence'].apply(clean_sequence)
    
    # Save the cleaned dataframe to a new csv file
    df.to_csv('cleaned_data.csv', index=False)
    
    print(f"Cleaned data is saved in the current directory: {os.getcwd()}/cleaned_data.csv")
    return df

# Specify the path of the file you want to load and clean
tsv_file_path = "/Users/andrejhric_1/git_projects/DataScience/Data_Science_Module_Project_predictor_of_cellular_compartment/predictor/training_input.tsv"

# Call the function to load and clean the data
df = load_and_clean_data(tsv_file_path)