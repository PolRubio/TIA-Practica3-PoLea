import argparse
from typing import Dict

import pandas as pd

from wineQualityModel import WineQualityModel
from wineQualityAnalysis import WineQualityAnalysis
from wineType import WineType

newWines: Dict[WineType, pd.DataFrame] = {
    WineType.Red: pd.DataFrame({
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
    }),
    WineType.White: pd.DataFrame({
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
}

def main(args):
    if args.operation == 'testModel':
        print("TESTING WINE QUALITY PREDICTION MODEL")
        wineModels: Dict[WineQualityModel] = {}
        wineModels[WineType.Red] = WineQualityModel(args.redWinePath, WineType.Red, args.testSize)
        wineModels[WineType.White] = WineQualityModel(args.whiteWinePath, WineType.White, args.testSize)

        for wineModel in wineModels.values():
            print("\n\n" + "="*50)
            print("\nWine type:", wineModel.wineType.name)
            wineModel.preprocessData()
            wineModel.splitData()
            wineModel.trainModel()
            accuracy, confMatrix, classReport = wineModel.evaluateModel()
            print("\n\tAccuracy: ", accuracy, "\n\tConfusion Matrix:\n", confMatrix, "\n\tClassification Report:\n", classReport)
            
            predictedQuality: int = wineModel.predictQuality(newWines[wineModel.wineType])
            print("\n\tPredicted quality for the new", wineModel.wineType.name, "wine sample: ", predictedQuality)


    elif args.operation == 'qualityAnalysis':
        print("ANALYZING WINE QUALITY")
        wineAnalyses: Dict[WineQualityAnalysis] = {}
        wineAnalyses[WineType.Red] = WineQualityAnalysis(args.redWinePath, WineType.Red)
        wineAnalyses[WineType.White] = WineQualityAnalysis(args.whiteWinePath, WineType.White)

        for wineAnalysis in wineAnalyses.values():
            print("\nWine type:", wineAnalysis.wineType.name)
            wineAnalysis.plotQualityDistribution()
            for quality in WineQualityAnalysis.Quality:
                wineAnalysis.plotQualityCharacteristics(quality)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Wine Quality Prediction')
    parser.add_argument('--operation', type=str, choices=['testModel', 'qualityAnalysis'], required=True, help='Operation to perform: testModel or qualityAnalysis')
    parser.add_argument('--redWinePath', type=str, help='Path to the red wine dataset', default='data/winequality/winequality-red.csv')
    parser.add_argument('--whiteWinePath', type=str, help='Path to the white wine dataset', default='data/winequality/winequality-white.csv')
    parser.add_argument('--testSize', type=float, help='Test size for train/test split', default=0.3)
    args = parser.parse_args()

    main(args)