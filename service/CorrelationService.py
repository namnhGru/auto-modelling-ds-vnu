from scipy import stats
from statsmodels.formula.api import ols
import statsmodels.api as sm
import pandas as pd
import model.Banner


class CorrelationService:

    @staticmethod
    def numbericToNumberic(numbericFeature1, numbericFeature2):
        cor, p_value = stats.pearsonr(numbericFeature1, numbericFeature2)
        return p_value

    @staticmethod
    def numbericToFactor(numbericFeature, factorFeature):

        def findVarName(obj, namespace=locals()):
            return [name for name in namespace if namespace[name] is obj][0]

        stringModel = findVarName(numbericFeature) + "~" + findVarName(factorFeature)
        lm_model = ols(stringModel, data=pd.concat([factorFeature, numbericFeature], axis=1)).fit()
        anova_table = sm.stats.anova_lm(lm_model)
        return anova_table['PR(>F)'][0]

    @staticmethod
    def factorToFactor(factorFeature1, factorFeature2):
        stringModel = "'" + factorFeature1 + "~" + factorFeature2 + "'"
        model = ols(stringModel).fit()
        anova_table = sm.stats.anova_lm(model)
        return anova_table

    @staticmethod
    def corForNumbericVariable(selectLinearFeatures):
        selectLinearFeaturesLength = len(selectLinearFeatures.columns)
        cor_data = []
        for i in range(0,selectLinearFeaturesLength-1):
            for j in range(i+1,selectLinearFeaturesLength):
                p_value = CorrelationService.numbericToNumberic(
                    selectLinearFeatures.iloc[:, i],
                    selectLinearFeatures.iloc[:, j]
                )
                if p_value < 0.05:
                    cor_data.append((selectLinearFeatures.iloc[:, i].name, selectLinearFeatures.iloc[:, j].name))
                    model.Banner.Banner.show(
                        "# Generate feature "
                        + selectLinearFeatures.iloc[:, i].name
                        + " * " + selectLinearFeatures.iloc[:, j].name
                        + " base on cor.test between features because p-value is "
                        + str(p_value)
                    )

        return cor_data

    @staticmethod
    def corLinearWithModelTarget(selectLinearFeatures, selectModelTarget):
        for _ in selectLinearFeatures.columns:
            p_value = CorrelationService.numbericToNumberic(
                selectLinearFeatures[_],
                selectModelTarget
            )
            if p_value > 0.05:
                selectLinearFeatures = selectLinearFeatures.drop(_, axis=1)
                model.Banner.Banner.show(
                    "# Drop feature "
                    + _
                    + " base on cor.test because p-value is "
                    + str(p_value)
                )

        return selectLinearFeatures


    @staticmethod
    def corFactorWithModelTarget(selectFactorFeatures, selectModelTarget):
        for _ in selectFactorFeatures.columns:
            p_value = CorrelationService.numbericToFactor(selectModelTarget, selectFactorFeatures[_])
            if p_value > 0.05:
                selectFactorFeatures = selectFactorFeatures.drop(_, axis=1)
                model.Banner.Banner.show(
                    "# Drop feature "
                    + _
                    + " base on ANOVA test because p-value is "
                    + str(p_value)
                )

        return selectFactorFeatures


