import pandas as pd
from model.GenData import GenData
from service.CorrelationService import CorrelationService


class AfterCorData:

    def __init__(self):
        self.genData = GenData()
        self.linearFeatureData = pd.DataFrame()
        self.factorFeatureData = pd.DataFrame()
        self.levelFeatureData = pd.DataFrame()
        self.modelTargetData = pd.DataFrame()
        self.corBetweenData = []
        self.newData = pd.DataFrame()

    def setGenData(self, genData):
        self.genData = genData

    def setLinearFeatureData(self):
        self.linearFeatureData = CorrelationService.corLinearWithModelTarget(
            self.genData.linearFeatureData,
            self.genData.modelTargetData
        )

    def setFactorFeatureData(self):
        self.factorFeatureData = CorrelationService.corFactorWithModelTarget(
            self.genData.factorFeatureData,
            self.genData.modelTargetData
        )

    def setLevelFeatureData(self):

        self.levelFeatureData = self.genData.levelFeatureData.copy()

    def setModelTargetData(self):
        self.modelTargetData = self.genData.modelTargetData.copy()

    def setCorBetweenData(self):
        self.corBetweenData = CorrelationService.corForNumbericVariable(
            self.linearFeatureData
        )

    def setNewData(self):
        self.newData = pd.concat([
            self.linearFeatureData,
            self.factorFeatureData,
            self.levelFeatureData
        ], axis=1)
