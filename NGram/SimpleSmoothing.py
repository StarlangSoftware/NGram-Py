from abc import abstractmethod


class SimpleSmoothing:

    @abstractmethod
    def setProbabilities(self,
                         nGram,
                         level: int):
        pass

    def setProbabilitiesGeneral(self, nGram):
        """
        Calculates the N-Gram probabilities with simple smoothing.
        :param nGram: N-Gram for which simple smoothing calculation is done.
        """
        self.setProbabilities(nGram, nGram.getN())
