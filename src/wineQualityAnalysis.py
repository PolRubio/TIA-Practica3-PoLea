from enum import Enum

import pandas as pd
import matplotlib.pyplot as plt

from wineType import WineType

class WineQualityAnalysis:
    """
    WineQualityAnalysis class to analyze and visualize the characteristics and distribution of wine quality.

    This class provides methods to:
    - Load the wine dataset.
    - Categorize wine quality into Low, Medium, and High.
    - Plot the distribution of wine quality categories.
    - Plot the average characteristics of wines in each quality category.
    """
    
    class Quality(Enum):
        Low = 0
        Medium = 1
        High = 2

    def __init__(self, winePath: str, wineType: WineType):
        self.winePath: str = winePath
        self.wineType: WineType = wineType
        self.wineData: pd.DataFrame = None
        self.loadData()
        self.categorizeQuality()

    def loadData(self) -> None:
        self.wineData = pd.read_csv(self.winePath, sep=';')

    def categorizeQuality(self) -> None:
        self.wineData['QualityCategory'] = pd.cut(self.wineData['quality'], bins=[0, 4, 6, 10], labels=[e.name for e in self.Quality])

    def plotQualityDistribution(self) -> None:
        qualityCounts = self.wineData['QualityCategory'].value_counts()
        qualityCounts.plot(kind='bar')
        plt.title('Distribution of Wine Quality for ' + self.wineType.name + ' Wine')
        plt.xlabel('Quality Category')
        plt.ylabel('Count')
        plt.show()

    def plotQualityCharacteristics(self, quality: Quality) -> None:
        qualityWines = self.wineData[self.wineData['QualityCategory'] == quality.name]

        avgValues: pd.Series = qualityWines.drop(columns=['quality', 'QualityCategory']).mean()
        avgValues.plot(kind='bar', figsize=(10, 6))
        plt.title('Average Values for ' + quality.name + '-Quality Wine for ' + self.wineType.name + ' Wine')
        plt.xlabel('Wine Attributes')
        plt.ylabel('Average Values')
        plt.xticks(rotation=45)
        plt.show()