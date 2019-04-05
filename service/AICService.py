from model.ModelProperties import ModelProperties
from model.AfterCorData import AfterCorData
from service.DistributionService import DistributionService
import pandas as pd


class AICService:
    def __init__(self):
        self.afterCorData = AfterCorData()
        self.distributionService = DistributionService()
        self.currentModelProperties = ModelProperties()
        self.nextModelProperties = ModelProperties()
        self.sortedPValues = pd.Series()
        self.currentAIC = {}
        self.nextDrop = ''
        self.nextAIC = {}
        self.minAIC = {}

    def setAfterCorData(self, afterCorData):
        self.afterCorData = afterCorData

    def setDistributionService(self, distributionService):
        self.distributionService = distributionService

    def setCurrentModelProperties(self, currentModelProperties):
        self.currentModelProperties = currentModelProperties

    def setSortedPValue(self):
        self.sortedPValues = self.currentModelProperties.resModel.pvalues.sort_values(ascending=False)

    def setCurrentAIC(self):
        self.currentAIC = {self.currentModelProperties.stringModel: self.currentModelProperties.resModel.aic}

    def setNextDrop(self):
        if len(self.sortedPValues) > 2:
            self.nextDrop = self.sortedPValues.index[0]
            self.sortedPValues.drop(index=self.nextDrop, inplace=True)

    def setNextModelProperties(self):
        self.nextModelProperties.stringModel = self.currentModelProperties.stringModel.replace(self.nextDrop, '', 1)
        self.nextModelProperties.stringModel = self.nextModelProperties.stringModel.replace('+', '')
        self.nextModelProperties.stringModel = self.nextModelProperties.stringModel.replace(' ', '')
        self.nextModelProperties.stringModel = self.nextModelProperties.stringModel.replace(']n', '] + n')
        self.nextModelProperties.modelType = self.currentModelProperties.modelType
        self.nextModelProperties.familyDistribution = self.currentModelProperties.familyDistribution
        self.nextModelProperties.setAfterCorData(self.afterCorData)
        self.nextModelProperties.setResModel()

    def setNextAIC(self):
        self.nextAIC = {self.nextModelProperties.stringModel: self.nextModelProperties.resModel.aic}

    def setMinAIC(self):
        if list(self.nextAIC.values())[0] == list(self.currentAIC.values())[0]:
            self.minAIC = self.nextAIC













