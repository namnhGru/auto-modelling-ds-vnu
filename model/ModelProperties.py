import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels.genmod.families.links as L


from model.Banner import Banner
from model.AfterCorData import AfterCorData
from service.InputController import InputController
from service.DistributionService import DistributionService


class ModelProperties:
    def __init__(self):
        self.inputController = InputController()
        self.afterCorData = AfterCorData()
        self.distributionService = DistributionService()
        self.modelType = ''
        self.familyDistribution = ''
        self.stringModel = ''
        self.resModel = ''

    def setInputController(self, inputController):
        self.inputController = inputController

    def setAfterCorData(self, afterCorData):
        self.afterCorData = afterCorData

    def setDistributionService(self, distributionService):
        self.distributionService = distributionService

    def setModelType(self):
        self.inputController.setModellingMethod()
        self.modelType = self.inputController.modellingMethod

    def setStringModel(self):
        stringModel = self.afterCorData.modelTargetData.name + " ~ "
        for _ in self.afterCorData.newData.columns:
            if _ == self.afterCorData.newData.columns[-1]:
                stringModel += 'newData' \
                               + "['" + _ + "']"
            else:
                stringModel += 'newData' \
                               + "['" + _ + "']" + " + "

        if len(self.afterCorData.corBetweenData) != 0:
            stringModel += " + "

        for item in self.afterCorData.corBetweenData:
            for _ in item:
                if _ == item[-1]:
                    stringModel += 'newData' \
                                   + "['" + _ + "']"
                else:
                    stringModel += 'newData' \
                                   + "['" + _ + "']" + ":"

            if item == self.afterCorData.corBetweenData[-1]:
                stringModel += ''
            else:
                stringModel += " + "

        Banner.show(self.stringModel)

        self.stringModel = stringModel

    def setFamilyDistribution(self):
        comparator = list(self.distributionService.selectedDistribution.keys())[0]
        if comparator == 'Normal':
            self.familyDistribution = sm.families.Gaussian(
                link=L.identity
            )
        if comparator == 'Poisson':
            self.familyDistribution = sm.families.Poisson(
                link=L.log
            )
        if comparator == 'Exp':
            self.familyDistribution = sm.families.Gaussian(
                link=L.inverse_power
            )
        if comparator == 'Gamma':
            self.familyDistribution = sm.families.Gamma(
                link=L.inverse_power
            )
        if comparator == 'Binomial':
            self.familyDistribution = sm.families.Binomial(
                link=L.logit
            )

    def setResModel(self):
        newData = self.afterCorData.newData
        if self.modelType == 'lm'.casefold():
            self.resModel = smf.ols(
                self.stringModel,
                data=pd.concat([
                    self.afterCorData.newData,
                    self.afterCorData.modelTargetData
                ],
                    axis=1
                )
            ).fit()

        if self.modelType == 'glm'.casefold():
            self.resModel = smf.glm(
                self.stringModel,
                data=pd.concat([
                    self.afterCorData.newData,
                    self.afterCorData.modelTargetData
                ],
                    axis=1
                ),
                family=self.familyDistribution
            ).fit()

        Banner.show("# Model Summary")
        print(self.resModel.summary())

