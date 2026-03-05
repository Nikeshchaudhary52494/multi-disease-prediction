import pandas as pd
import numpy as np


def prepare_symptoms_array(symptoms):
    '''
    Convert a list of symptoms to a DataFrame with named columns that matches
    the dataframe used to train the machine learning model.

    Output:
    - X (pd.DataFrame) = X values ready as input to ML model to get prediction
    '''
    df = pd.read_csv('data/clean_dataset.tsv', sep='\t')
    feature_columns = df.columns[:-1]  # all columns except the last (target)

    symptoms_array = np.zeros((1, len(feature_columns)))
    symptoms_df = pd.DataFrame(symptoms_array, columns=feature_columns)

    for symptom in symptoms:
        if symptom in symptoms_df.columns:
            symptoms_df[symptom] = 1

    return symptoms_df