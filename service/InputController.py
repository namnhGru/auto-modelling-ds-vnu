
class InputController:
    def __init__(self):
        self.originalData = ''
        self.linearFeatureBeginPoint = '0'
        self.linearFeatureEndPoint = '0'
        self.factorFeatureBeginPoint = '0'
        self.factorFeatureEndPoint = '0'
        self.levelFeatureBeginPoint = '0'
        self.levelFeatureEndPoint = '0'
        self.modelTarget = ''
        self.confidentInterval = 0
        self.modellingMethod = ''

    def setOriginalData(self):
        self.originalData = input("Data to Analyze is: ")

    def setLinearFeaturePoint(self):
        try:
            self.linearFeatureBeginPoint, self.linearFeatureEndPoint = input("Column to use as Linear Feature (a->b, "
                                                                             "leave if none): ").split('->')
        except ValueError:
            pass

    def setFactorFeaturePoint(self):
        try:
            self.factorFeatureBeginPoint, self.factorFeatureEndPoint = input("Column to use as Factor Feature (a->b, "
                                                                             "leave if none): ").split('->')
        except ValueError:
            pass

    def setLevelFeaturePoint(self):
        try:
            self.levelFeatureBeginPoint, self.levelFeatureEndPoint = input("Column to use as Level Feature (a->b, "
                                                                           "leave if none): ").split('->')
        except ValueError:
            pass

    def setModelTarget(self):
        self.modelTarget = input("Column is used as Model Target: ")

    def setConfidentInterval(self):
        self.confidentInterval = input("Confident Interval: ")

    def setModellingMethod(self):
        self.modellingMethod = input("Please choose Modelling method (lm/ glm): ")

    @staticmethod
    def setLevelForEachFeature(feature):
        return input("Order of Level Data " + feature + ": ")



