from abc import abstractmethod
from NGram.NGram import NGram


class SimpleSmoothing:

    @abstractmethod
    def setProbabilities(self, nGram: NGram, level: int):
        pass

    def setProbabilitiesGeneral(self, nGram: NGram):
        self.setProbabilities(nGram, nGram.getN())