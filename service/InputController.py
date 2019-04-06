
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

    @staticmethod
    def checkPoint(point):
        if point == ['']:
            a = '0'
            b = '0'
        else:
            if len(point) == 2:
                a = point[0]
                b = point[1]
            else:
                a = point[0]
                b = point[0]

        return a, b

    def setOriginalData(self):
        self.originalData = input("Data to Analyze is: ")

    def setLinearFeaturePoint(self):
        point = input("Column to use as Linear Feature (a->b, leave if none): ").split('->')
        self.linearFeatureBeginPoint, self.linearFeatureEndPoint = InputController.checkPoint(point)

    def setFactorFeaturePoint(self):
        point = input("Column to use as Factor Feature (a->b, leave if none): ").split('->')
        self.factorFeatureBeginPoint, self.factorFeatureEndPoint = InputController.checkPoint(point)

    def setLevelFeaturePoint(self):
        point = input("Column to use as Level Feature (a->b, leave if none): ").split('->')
        self.levelFeatureBeginPoint, self.levelFeatureEndPoint = InputController.checkPoint(point)

    def setModelTarget(self):
        self.modelTarget = input("Column is used as Model Target: ")

    def setConfidentInterval(self):
        self.confidentInterval = input("Confident Interval: ")

    def setModellingMethod(self):
        self.modellingMethod = input("Please choose Modelling method (lm/ glm): ")

    @staticmethod
    def setLevelForEachFeature(feature):
        return input("Order of Level Data " + feature + ": ")



