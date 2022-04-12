"""
This class provides techniques to transform the data, including normalization
and transform the input data, etc. 
"""
import numpy as np
from typing import List

class DataNormalization:
    def __init__(self):
        """
        Initialization
        """
        self.mean_F_db = None
        self.sigma_F_db = None
        self.mean_R_db = None
        self.sigma_R_db = None

    def fit(self, F_train: np.array, R_train: np.array) -> None:
        """
        Fit the train data includes F_train and R_train. The mean and sigma
        values of F_train and R_train will be assigned into the object.
        """
        print("Fitting the input data")
        # processing the data
        # Convert to dB
        F_train_db = 10.0 * np.log10(F_train)
        R_train_db = 10.0 * np.log10(R_train)
        # Get mean and sigma of F_train_db
        mean_F_db = np.mean(F_train_db)
        sigma_F_db = np.sqrt(np.var(F_train_db))

        # Get mean and sigma of R_train_db
        mean_R_db = np.mean(R_train_db)
        sigma_R_db = np.sqrt(np.var(R_train_db))

        # Assign value into the object
        print("Assigning mean and sigma values")
        self.mean_F_db = mean_F_db
        self.sigma_F_db = sigma_F_db
        self.mean_R_db = mean_R_db
        self.sigma_R_db = sigma_R_db
        print("Done")

    def transform(self, F: np.array, R: np.array) -> List:
        """
        Transform the input data F and R using the mean, sigma calculated
        from calling fit method.

        Args:
            F (np.array): [Input data]
            R (np.array): [Input data]

        Returns:
            (list): [The transformed data]
        """
        return (10.0 * np.log10(F) - self.mean_F_db) / self.sigma_F_db, (
            10.0 * np.log10(R) - self.mean_R_db
        ) / self.sigma_R_db
