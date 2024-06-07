import argparse
import pandas as pd
import numpy as np
from enum import Enum
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from typing import Tuple, Literal
import matplotlib.pyplot as plt

class WineQualityModel:
    def __init__(self, winePath: str, wineType: str, testSize: float = 0.3):
        self.wine = pd.read_csv(winePath, sep=';')
        self.wineType = wineType
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

        print("Model accuracy for", self.wineType, "wine:", accuracy)
        print("Confusion matrix for", self.wineType, "wine:\n", confMatrix)
        print("Classification report for", self.wineType, "wine:\n", classReport)

    def predictQuality(self, newSample: pd.DataFrame) -> int:
        return self.model.predict(newSample)[0]

class WineQualityAnalysis:
    class Quality(Enum):
        Low = 0
        Medium = 1
        High = 2

    def __init__(self, winePath: str, wineType: str):
        self.winePath = winePath
        self.wineType = wineType
        self.wineData = None

    def loadData(self) -> None:
        self.wineData = pd.read_csv(self.winePath, sep=';')

    def categorizeQuality(self) -> None:
        self.wineData['QualityCategory'] = pd.cut(self.wineData['quality'], bins=[0, 4, 6, 10], labels=[e.name for e in self.Quality])

    def plotQualityDistribution(self) -> None:
        self.loadData()
        self.categorizeQuality()
        qualityCounts = self.wineData['QualityCategory'].value_counts()
        qualityCounts.plot(kind='bar')
        plt.title('Distribution of Wine Quality for ' + self.wineType + ' Wine')
        plt.xlabel('Quality Category')
        plt.ylabel('Count')
        plt.show()

    def plotQualityCharacteristics(self, quality: Quality) -> None:
        self.loadData()
        self.categorizeQuality()
        qualityWines = self.wineData[self.wineData['QualityCategory'] == quality.name]

        features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                    'pH', 'sulphates', 'alcohol']

        featuresColumns = qualityWines[features]

        avgValues = featuresColumns.mean()

        avgValues.plot(kind='bar', figsize=(10, 6))
        plt.title('Average Values for ' + quality.name + '-Quality Wine for ' + self.wineType + ' Wine')
        plt.xlabel('Wine Attributes')
        plt.ylabel('Average Values')
        plt.xticks(rotation=45)
        plt.show()

def main(args):
    # Create instances for red and white wine models
    redWineModel = WineQualityModel(args.redWinePath, args.testSize, 'Red')
    whiteWineModel = WineQualityModel(args.whiteWinePath, args.testSize, 'White')

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
    args = parser.parse_args()

    # main(args)

    # Data analysis for wine quality
    wineAnalysisRed = WineQualityAnalysis(args.redWinePath, 'Red')
    wineAnalysisWhite = WineQualityAnalysis(args.whiteWinePath, 'White')

    # Plot quality distribution
    wineAnalysisRed.plotQualityDistribution()
    wineAnalysisWhite.plotQualityDistribution()

    # Plot quality characteristics
    for quality in WineQualityAnalysis.Quality:
        wineAnalysisRed.plotQualityCharacteristics(quality)
        wineAnalysisWhite.plotQualityCharacteristics(quality)
