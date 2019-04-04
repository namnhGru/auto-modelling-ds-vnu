class Banner:
    def __init__(self, headerBanner, inputBanner, resultBanner, predictBanner):
        self.headerBanner = headerBanner
        self.inputBanner = inputBanner
        self.resultBanner = resultBanner
        self.predictBanner = predictBanner

    def showHeaderBanner(self):
        textLength = len(self.headerBanner)
        boxWidth = textLength + 20
        print("")
        print("")
        print("|" + "="*boxWidth + "|")
        print("|" + " "*boxWidth + "|")
        print("|" + " "*5 + "-"*5 + self.headerBanner + "-"*5 + " "*5 + "|")
        print("|" + " "*boxWidth + "|")
        print("|" + "="*boxWidth + "|")
        print("")
        print("")

    def showInputBanner(self):
        textLength = len(self.inputBanner)
        boxWidth = round((20-textLength)/2)
        print("")
        print("# " + "="*boxWidth + self.inputBanner + "="*boxWidth + " #")
        print("")

    def showResultBanner(self):
        textLength = len(self.resultBanner)
        boxWidth = round((20-textLength)/2)
        print("")
        print("# " + "="*boxWidth + self.resultBanner + "="*boxWidth + " #")
        print("")

    def showPredictBanner(self):
        textLength = len(self.predictBanner)
        boxWidth = round((20-textLength)/2)
        print("")
        print("# " + "="*boxWidth + self.predictBanner + "="*boxWidth + " #")
        print("")

    @staticmethod
    def show(toShow):
        print("")
        print(toShow)
        print("")
