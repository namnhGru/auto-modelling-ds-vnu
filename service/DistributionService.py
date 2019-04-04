from scipy import stats
from skgof import cvm_test, ks_test
import numpy as np
from model.Banner import Banner
from model.GenData import GenData


class DistributionService:

    def __init__(self):
        self.genData = GenData()
        self.selectedDistribution = ''
        self.passedDistribution = {}
        self.failedDistribution = {}
        self.DistributionScoreNormal = {}
        self.DistributionScorePoisson = {}
        self.DistributionScoreExp = {}
        self.DistributionScoreGamma = {}
        self.DistributionScoreBinomial = {}

    def setGenData(self, genData):
        self.genData = genData

    def setNormalDistributionScore(self):
        W, p_value = stats.shapiro(self.genData.modelTargetData)

        if p_value < 0.05:
            self.DistributionScoreNormal = {'Normal': (False, p_value)}
        else:
            self.DistributionScoreNormal = {'Normal': (True, p_value)}

    def setPoissonDistributionScore(self):
        meanOfData = self.genData.modelTargetData.mean()
        expectedPoissonDistributionData = stats.poisson(meanOfData)
        statistic, p_value = cvm_test(
            self.genData.modelTargetData,
            expectedPoissonDistributionData
        )
        if p_value < 0.05:
            self.DistributionScorePoisson = {'Poisson': (False, p_value)}
        else:
            self.DistributionScorePoisson = {'Poisson': (True, p_value)}

    def setExpDistributionScore(self):
        meanOfData = self.genData.modelTargetData.mean()
        expectedExpDistributionData = stats.expon(1 / meanOfData)
        statistic, p_value = ks_test(
            self.genData.modelTargetData,
            expectedExpDistributionData
        )

        if p_value < 0.05:
            self.DistributionScoreExp = {'Exp': (False, p_value)}
        else:
            self.DistributionScoreExp = {'Exp': (True, p_value)}

    def setGammaDistributionScore(self):
        alpha = self.genData.modelTargetData.mean() ** 2 / np.var(self.genData.modelTargetData)
        statistic, p_value = stats.kstest(
            self.genData.modelTargetData,
            cdf='gamma',
            args=(
                alpha,
            )
        )

        if p_value < 0.05:
            self.DistributionScoreGamma = {'Gamma': (False, p_value)}
        else:
            self.DistributionScoreGamma = {'Gamma': (True, p_value)}

    def setBinomialDistrbutionScore(self):
        sizeOfDataSeries = self.genData.modelTargetData.size
        freqDataSeries, binDataSeries = np.histogram(self.genData.modelTargetData, bins=2)
        probabilityOfTest_01, probabilityOfTest_02 = freqDataSeries / sizeOfDataSeries
        expectedBinomialDistributionData_01 = stats.binom(
            sizeOfDataSeries,
            probabilityOfTest_01,
        )
        expectedBinomialDistributionData_02 = stats.binom(
            sizeOfDataSeries,
            probabilityOfTest_02
        )

        statistic_01, p_valueBinomial_01 = cvm_test(
            self.genData.modelTargetData,
            expectedBinomialDistributionData_01
        )
        statistic_01, p_valueBinomial_02 = cvm_test(
            self.genData.modelTargetData,
            expectedBinomialDistributionData_02
        )

        if p_valueBinomial_01 < 0.05 or p_valueBinomial_02 < 0.05:
            self.DistributionScoreBinomial = {'Binomial': (False, max(p_valueBinomial_01, p_valueBinomial_02))}
        else:
            self.DistributionScoreBinomial = {'Binomial': (True, min(p_valueBinomial_01, p_valueBinomial_02))}

    def setPassedDistribution(self):
        for _ in [
            self.DistributionScoreNormal,
            self.DistributionScorePoisson,
            self.DistributionScoreExp,
            self.DistributionScoreGamma,
            self.DistributionScoreBinomial
        ]:
            if list(_.values())[0][0] is True:
                self.passedDistribution.update({list(_.keys())[0]: list(_.values())[0][1]})

    def setFailedDistribution(self):
        for _ in [
            self.DistributionScoreNormal,
            self.DistributionScorePoisson,
            self.DistributionScoreExp,
            self.DistributionScoreGamma,
            self.DistributionScoreBinomial
        ]:
            if list(_.values())[0][0] is False:
                self.failedDistribution.update({list(_.keys())[0]: list(_.values())[0][1]})

    def setSelectedDistribution(self):
        self.selectedDistribution = {
            max(self.passedDistribution, key=self.passedDistribution.get):
            max(self.passedDistribution.values())
        }

        Banner.show(
            "# Distribution of Model Target is "
            + str(self.selectedDistribution.keys())
            + " because of p-value is "
            + str(self.selectedDistribution.values())
            + " when do selecting between "
            + str(len(self.passedDistribution))
            + " passed distributions (name: p-value): "
            + str(self.passedDistribution)
        )

        Banner.show(
            "# "
            + str(len(self.failedDistribution))
            + " false distribution are (name: p-value): "
            + str(self.failedDistribution)
        )
