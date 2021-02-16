import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import os.path


def load_data():
    root_dir = '..'
    fpath = os.path.join(root_dir, 'data/adult.csv')

    data = pd.read_csv(fpath, na_values='?')

    # drop duppl rows
    data = data.drop_duplicates().reset_index(drop=True)
    
    # drop useless columns
    data = data.drop(columns=[
        'fnlwgt','race','native.country','education','relationship'
    ])

    target = 'income'

    data[target] = data[target].replace({
        '<=50K':0,
        '>50K':1
    })

    return data


def load_preprocessor():
    numeric_features = ['age', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_features = ['workclass', 'marital.status', 'occupation', 'sex']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    return preprocessor

def load_data_preprocess(preprocessor, data):
    preprocessor = preprocessor.fit(data)
    data_preprocessed = preprocessor.transform(data)

    return data_preprocessed, preprocessor
