from abc import abstractmethod


class SimpleSmoothing:

    @abstractmethod
    def setProbabilities(self, nGram, level: int):
        pass

    def setProbabilitiesGeneral(self, nGram):
        self.setProbabilities(nGram, nGram.getN())
