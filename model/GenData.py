from service.InputController import InputController
import pandas as pd


class GenData:

    def __init__(self):
        self.inputService = InputController()
        self.originalData = pd.DataFrame()
        self.linearFeatureData = pd.DataFrame()
        self.factorFeatureData = pd.DataFrame()
        self.levelFeatureData = pd.DataFrame()
        self.modelTargetData = pd.DataFrame()
        self.newData = pd.DataFrame()

    def setInputService(self, inputService):
        self.inputService = inputService

    def setOriginalData(self):
        self.inputService.setOriginalData()
        self.originalData = pd.read_csv(self.inputService.originalData)

    def setLinearFeatureData(self):
        self.inputService.setLinearFeaturePoint()
        if self.inputService.linearFeatureBeginPoint == '0':
            return
        self.linearFeatureData = self.originalData.loc[
                                 :,
                                 self.inputService
                                     .linearFeatureBeginPoint:
                                 self.inputService
                                     .linearFeatureEndPoint
                                 ]

    def setFactorFeatureData(self):
        self.inputService.setFactorFeaturePoint()
        if self.inputService.factorFeatureBeginPoint == '0':
            return
        self.factorFeatureData = self.originalData.loc[
                                 :,
                                 self.inputService
                                     .factorFeatureBeginPoint:
                                 self.inputService
                                     .factorFeatureEndPoint
                                 ].astype('category')

    def setLevelFeatureData(self):
        self.inputService.setLevelFeaturePoint()
        if self.inputService.levelFeatureBeginPoint == '0':
            return
        self.levelFeatureData = self.originalData.loc[
                                :,
                                self.inputService
                                    .levelFeatureBeginPoint:
                                self.inputService
                                    .levelFeatureEndPoint
                                ]

    def setModelTarget(self):
        self.inputService.setModelTarget()
        self.modelTargetData = self.originalData.loc[
                               :,
                               self.inputService
                                   .modelTarget
                               ]

    def setLevelForEachLevelFeatureData(self):
        self.levelFeatureData = self.levelFeatureData.copy()
        self.levelFeatureData = self.levelFeatureData.astype('str')
        for _ in self.levelFeatureData.columns:
            orderCategory = InputController.setLevelForEachFeature(_)
            cat_dtype = pd.CategoricalDtype(
                categories=orderCategory.split(','),
                ordered=True
            )
            self.levelFeatureData[_] = self.levelFeatureData[_].astype(cat_dtype)

    def setNewData(self):
        self.newData = pd.concat([
            self.linearFeatureData,
            self.factorFeatureData,
            self.levelFeatureData
        ], axis=1)

    def __str__(self):
        return str(self.originalData) \
               + str('\n') \
               + str(self.linearFeatureData) \
               + str('\n') \
               + str(self.factorFeatureData) \
               + str('\n') \
               + str(self.levelFeatureData) \
               + str('\n') \
               + str(self.modelTargetData) \
               + str('\n')
