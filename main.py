from model.Banner import Banner
from model.GenData import GenData
from model.AfterCorData import AfterCorData
from model.ModelProperties import ModelProperties
from service.InputController import InputController
from service.DistributionService import DistributionService
from service.AICService import AICService
from service.PredictService import PredictService
import matplotlib.pyplot as plt

#################################
#                               #
#           Init Data           #
#                               #
#################################
# DATA_SOURCE = './data/normal.csv'
productBanner = Banner(
        headerBanner="AUTO MODELLING FOR STATISTIC",
        inputBanner="INPUT INFO",
        resultBanner="RESULT",
        predictBanner="PREDICT"
    )

#################################
#                               #
#        Product Logic          #
#                               #
#################################
productBanner.showHeaderBanner()
productBanner.showInputBanner()

inputController = InputController()

genData = GenData()
genData.setInputService(inputController)
genData.setOriginalData()
print(genData.originalData.head(5))
genData.setLinearFeatureData()
genData.setFactorFeatureData()
# genData.setLevelFeatureData()
# genData.setLevelForEachLevelFeatureData()
genData.setModelTarget()
genData.setNewData()
print(genData.newData)
productBanner.showResultBanner()
productBanner.show(genData.newData.head(5))

afterCorData = AfterCorData()
afterCorData.setGenData(genData)
afterCorData.setLinearFeatureData()
# afterCorData.setLevelFeatureData()
afterCorData.setFactorFeatureData()
afterCorData.setModelTargetData()
afterCorData.setCorBetweenData()
afterCorData.setNewData()

productBanner.showResultBanner()
productBanner.show(afterCorData.newData.head(5))

selectedDistribution = DistributionService()
selectedDistribution.setGenData(genData)
selectedDistribution.setBinomialDistrbutionScore()
selectedDistribution.setNormalDistributionScore()
selectedDistribution.setExpDistributionScore()
selectedDistribution.setPoissonDistributionScore()
selectedDistribution.setGammaDistributionScore()
selectedDistribution.setPassedDistribution()
selectedDistribution.setFailedDistribution()
selectedDistribution.setSelectedDistribution()

modelProperties = ModelProperties()
modelProperties.setAfterCorData(afterCorData)
modelProperties.setDistributionService(selectedDistribution)
modelProperties.setFamilyDistribution()
modelProperties.setStringModel()
productBanner.showResultBanner()
modelProperties.setModelType()
productBanner.showResultBanner()
modelProperties.setResModel()

productBanner.showResultBanner()
aicService = AICService()
aicService.setAfterCorData(afterCorData)
aicService.setCurrentModelProperties(modelProperties)
aicService.setSortedPValue()
aicService.setCurrentAIC()
aicService.setDistributionService(selectedDistribution)

while aicService.minAIC == {}:
    aicService.setNextDrop()
    aicService.setNextModelProperties()
    aicService.setNextAIC()
    aicService.setMinAIC()

    Banner.show("*** Current Model & AIC: " + str(aicService.currentAIC))
    Banner.show("*** Next Model & AIC: " + str(aicService.nextAIC))
    Banner.show("*** Min Model & AIC: " + str(aicService.minAIC))

    aicService.setCurrentModelProperties(aicService.nextModelProperties)
    aicService.setSortedPValue()
    aicService.setCurrentAIC()

productBanner.showPredictBanner()
predictService = PredictService()
predictService.setInputController(inputController)
predictService.setModelProperties(modelProperties)
predictService.setInputData()
predictService.setTestData()
predictService.setTrainResult()
predictService.setTestResult()
predictService.setTrainScore()
predictService.setTestScore()

print("Train Score: " + str(round(predictService.trainScore*100, 2)))
print("Test Score: " + str(round(predictService.testScore*100, 2)))

plt.subplot(2, 1, 1)
plt.title("Train Score: " + str(round(predictService.trainScore*100, 2)))
plt.plot(genData.modelTargetData, label='real')
plt.plot(predictService.trainResult, label='predict')
plt.subplot(2, 1, 2)
plt.title("Test Score: " + str(round(predictService.testScore*100, 2)))
plt.plot(predictService.testData[modelProperties.afterCorData.modelTargetData.name])
plt.plot(predictService.testResult)
plt.show()

#
# generatedModel = ModellingService.lmOLSModelling(
#     frame_data,
#     frame_modelTarget,
#     afterCorrelationTest_frame_linearFeatures,
#     afterCorrelationTest_frame_linearFeatures_correlateBetween
# )




