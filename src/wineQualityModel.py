from typing import Tuple

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from wineType import WineType

class WineQualityModel:
    """
    WineQualityModel class to train and evaluate a wine quality prediction model.

    Machine Learning Method Used:
    K-Nearest Neighbors (KNN) Classifier:
    The KNN algorithm is used for classification by finding the most common class among the K nearest neighbors of a data point. 
    It is simple, intuitive, and effective for small to medium-sized datasets. 
    KNN is non-parametric, meaning it makes no assumptions about the underlying data distribution, 
    which is useful for datasets where the relationship between features and labels is complex or unknown.

    Alternative Method:
    Random Forest Classifier:
    Random Forest is an ensemble learning method that builds multiple decision trees and merges their results to improve accuracy and control overfitting. 
    It works by randomly selecting subsets of data and features, training individual trees, and then combining their predictions.
    This method is robust to noise and can handle large datasets with high dimensionality.

    How it works:
    - Randomly sample data and features to build multiple decision trees.
    - Each tree independently predicts the class of a data point.
    - The final prediction is determined by majority voting from all the trees.
    Random Forest offers high accuracy, handles overfitting better, and provides feature importance, which helps in understanding the data.
    """
    
    def __init__(self, winePath: str, wineType: WineType, testSize: float = 0.3):
        self.wine: pd.DataFrame = pd.read_csv(winePath, sep=';')
        self.wineType: WineType = wineType
        self.testSize: float = testSize
        self.model: KNeighborsClassifier = KNeighborsClassifier()

    def transformQuality(self, df: pd.DataFrame) -> pd.DataFrame:
        df['quality'] = df['quality'].apply(lambda x: 0 if x <= 4 else 1 if x <= 6 else 2)
        return df

    def preprocessData(self) -> None:
        self.wine = self.transformQuality(self.wine)

    def splitData(self) -> None:
        X = self.wine.drop('quality', axis=1)
        y = self.wine['quality']
        
        self.XTrain, self.XTest, self.yTrain, self.yTest = train_test_split(
            X, y, test_size=self.testSize)

    def trainModel(self) -> None:
        self.model.fit(self.XTrain, self.yTrain)

    def evaluateModel(self) -> Tuple[float, np.ndarray, str]:
        yPred = self.model.predict(self.XTest)
        accuracy = accuracy_score(self.yTest, yPred)
        confMatrix = confusion_matrix(self.yTest, yPred)
        classReport = classification_report(self.yTest, yPred)
        return accuracy, confMatrix, classReport

    def predictQuality(self, newSample: pd.DataFrame) -> int:
        return self.model.predict(newSample)[0]