import pandas as pd
from model.ModelProperties import ModelProperties
from service.InputController import InputController
from sklearn.metrics import r2_score


class PredictService:
    def __init__(self):
        self.inputController = InputController()
        self.modelProperties = ModelProperties()
        self.inputData = pd.DataFrame()
        self.testData = pd.DataFrame()
        self.trainResult = pd.Series()
        self.testResult = pd.Series()
        self.trainScore = 0
        self.testScore = 0

    def setInputController(self, inputController):
        self.inputController = inputController

    def setModelProperties(self, modelProperties):
        self.modelProperties = modelProperties

    def setInputData(self):
        self.inputData = pd.read_csv(self.inputController.originalData)

    def setTestData(self):
        self.testData = self.inputData.head(int(round(len(self.inputData) * 0.2)))

    def setTrainResult(self):
        self.trainResult = self.modelProperties.resModel.predict()

    def setTestResult(self):
        self.testResult = self.modelProperties.resModel.predict(exog={'newData': self.testData})

    def setTrainScore(self):
        self.trainScore = r2_score(
            y_true=self.modelProperties.afterCorData.modelTargetData,
            y_pred=self.trainResult
        )

    def setTestScore(self):
        self.testScore = r2_score(
            y_true=self.testData[self.modelProperties.afterCorData.modelTargetData.name],
            y_pred=self.testResult
        )








