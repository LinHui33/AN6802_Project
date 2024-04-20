# get_dummies.py
# Save your custom function in a Python script (.py file) then import it
# for to use it with pickle.load().
# This is a common approach to store and reuse custom functions in different scripts or projects.
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class GetDummies(BaseEstimator, TransformerMixin):
    def __init__(self, data_sep=',', col_name_sep='_'):
        """
        Transformer that creates dummy variables from categorical columns with a separator.
        Parameters:
            - data_sep (str): Defaulth ',' separator used to split categorical values into multiple dummy variables.
            - col_name_sep (str): Defaulth '_' separator used to separate the column name from the prefix in the output column names.
        """
        self.data_sep     = data_sep
        self.col_name_sep = col_name_sep
        self.columns      = []
        self.dummy_cols   = []
        self.dummy_prefix = []
        
    # Return self nothing else to do here
    def fit(self, X, y  = None): 
        """
        Fit the transformer to the data.
        Parameters:
            - X (pandas.DataFrame): Input data with categorical columns.
            - y (array-like): Target variable (ignored).
        Returns:
            - self: Returns the transformer object.
        """        
        object_cols       = X.select_dtypes(include="O").columns
        self.dummy_cols   = [col for col in object_cols if X[col].str.contains(self.data_sep, regex=True).any()]
        self.dummy_prefix = [col[:2] if self.col_name_sep not in col else ''.join(map(lambda x: x[0], col.split(self.col_name_sep))) for col in self.dummy_cols]
        
        if len(self.dummy_cols):
            # Apply dummy to train data
            dummy_frames_df = pd.concat([X[col].str.get_dummies(sep=self.data_sep).add_prefix(pre+self.col_name_sep) for pre, col in zip(self.dummy_prefix, self.dummy_cols)], axis=1)
            self.columns    = X.join(dummy_frames_df).drop(columns=self.dummy_cols).columns.tolist()
        else:
            self.columns = X.columns.tolist()
        return self
    
    # Transformer method for to return transformed data
    def transform(self, X, y = None):
        """
        Transform the input data by creating dummy variables.
        Parameters:
            - X (pandas.DataFrame): Input data with categorical columns.
            - y (array-like): Target variable (ignored).
        Returns:
            - X_transformed (pandas.DataFrame): Transformed data with dummy variables.
        """
        if len(self.dummy_prefix) or len(self.dummy_cols):
            dummy_frames_df = pd.concat([X[col].str.get_dummies(sep=self.data_sep).add_prefix(pre+self.col_name_sep) for pre, col in zip(self.dummy_prefix, self.dummy_cols)], axis=1)
            X = X.join(dummy_frames_df).reindex(columns=self.columns, fill_value=0)
        return X
        
    # to get feature names    
    def get_feature_names_out(self, input_features=None):
        """
        Get the names of the transformed features.
        Parameters:
            - input_features (array-like): Names of the input features (ignored).
        Returns:
            - output_features (list): Names of the transformed features.
        """
        return self.columns
