from sklearn.impute import KNNImputer
import numpy as np
import pandas as pd 

class Imputation:
    """
    This class is to impute missing values in the dataframe
    impute_method can be mean, mode, median, knn
    cols list of columns in the dataframe
    """
    def __init__(
        self,
        df,
        random_state = 42,
        save = True,
        ):
        self.df = df
        self.random_state = random_state
        self.save = save

    def impute(self, impute_method, cols):
        self.cols = cols
        self.impute_method = impute_method
        if self.impute_method == "mean":
            for c in self.cols:
                self.df[c].fillna(self.df[c].mean(), inplace=True)
        elif self.impute_method == "median":
            for c in self.cols:
                self.df[c].fillna(self.df[c].median(), inplace=True)
        elif self.impute_method == "mode":
            for c in self.cols:
                self.df[c].fillna(self.df[c].mode()[0], inplace=True)
        

        return self.df



