from model.Banner import Banner
from service.InputController import InputController
from model.GenData import GenData
from model.AfterCorData import AfterCorData
from model.ModelProperties import ModelProperties
from service.DistributionService import DistributionService
from service.AICService import AICService

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
genData.setLinearFeatureData()
genData.setFactorFeatureData()
# genData.setLevelFeatureData()
genData.setLevelForEachLevelFeatureData()
genData.setModelTarget()
genData.setNewData()

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


#
# generatedModel = ModellingService.lmOLSModelling(
#     frame_data,
#     frame_modelTarget,
#     afterCorrelationTest_frame_linearFeatures,
#     afterCorrelationTest_frame_linearFeatures_correlateBetween
# )




