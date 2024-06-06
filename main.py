import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from typing import Tuple

class WineQualityModel:
    def __init__(self, winePath: str, testSize: float = 0.3):
        self.wine = pd.read_csv(winePath, sep=';')
        self.testSize = testSize
        self.model = KNeighborsClassifier()

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

    def evaluate(self) -> None:
        accuracy, confMatrix, classReport = self.evaluateModel()

        print("Model accuracy: ", accuracy)
        print("Confusion matrix:\n", confMatrix)
        print("Classification report:\n", classReport)

    def predictQuality(self, newSample: pd.DataFrame) -> int:
        return self.model.predict(newSample)[0]

def main(args):
    # Create instances for red and white wine
    redWineModel = WineQualityModel(args.redWinePath, args.testSize)
    whiteWineModel = WineQualityModel(args.whiteWinePath, args.testSize)

    # Preprocess data
    redWineModel.preprocessData()
    whiteWineModel.preprocessData()

    # Data analysis (optional, not encapsulated in class)
    redQualityCounts = redWineModel.wine['quality'].value_counts()
    whiteQualityCounts = whiteWineModel.wine['quality'].value_counts()
    print("Red wine quality counts:\n", redQualityCounts)
    print("White wine quality counts:\n", whiteQualityCounts)

    redQualityMeans = redWineModel.wine.groupby('quality').mean()
    whiteQualityMeans = whiteWineModel.wine.groupby('quality').mean()
    print("Red wine quality means:\n", redQualityMeans)
    print("White wine quality means:\n", whiteQualityMeans)

    # Split data
    redWineModel.splitData()
    whiteWineModel.splitData()

    # Train models
    redWineModel.trainModel()
    whiteWineModel.trainModel()

    # Evaluate models
    print("Evaluating red wine model:")
    redWineModel.evaluate()
    
    print("Evaluating white wine model:")
    whiteWineModel.evaluate()

    # Predict the quality of new wine samples
    newRedWine = pd.DataFrame({
        'fixed acidity': [7.4],
        'volatile acidity': [1.015],
        'citric acid': [0.01],
        'residual sugar': [2.9],
        'chlorides': [0.075],
        'free sulfur dioxide': [4],
        'total sulfur dioxide': [11],
        'density': [0.896],
        'pH': [3.43],
        'sulphates': [0.5],
        'alcohol': [11.01]
    })

    newWhiteWine = pd.DataFrame({
        'fixed acidity': [5.7],
        'volatile acidity': [0.19],
        'citric acid': [0.28],
        'residual sugar': [10.6],
        'chlorides': [0.028],
        'free sulfur dioxide': [48],
        'total sulfur dioxide': [134],
        'density': [0.894],
        'pH': [3.02],
        'sulphates': [0.49],
        'alcohol': [11.03]
    })

    predictedRedQuality = redWineModel.predictQuality(newRedWine)
    predictedWhiteQuality = whiteWineModel.predictQuality(newWhiteWine)

    print("Predicted quality for the new red wine sample: ", predictedRedQuality)
    print("Predicted quality for the new white wine sample: ", predictedWhiteQuality)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wine Quality Prediction')
    parser = argparse.ArgumentParser(description='Wine Quality Prediction')
    parser.add_argument('--redWinePath', type=str, help='Path to the red wine dataset', default='winequality/winequality-red.csv')
    parser.add_argument('--whiteWinePath', type=str, help='Path to the white wine dataset', default='winequality/winequality-white.csv')
    parser.add_argument('--testSize', type=float, help='Test size for train/test split', default=0.3)
    parser.add_argument('--randomState', type=int, help='Random state for reproducibility', default=42)
    args = parser.parse_args()
    main(args)
