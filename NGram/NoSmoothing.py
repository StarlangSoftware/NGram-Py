from NGram.NGram import NGram
from NGram.SimpleSmoothing import SimpleSmoothing


class NoSmoothing(SimpleSmoothing):

    def setProbabilities(self,
                         nGram: NGram,
                         level: int):
        """
        Calculates the N-Gram probabilities with no smoothing
        :param nGram: N-Gram for which no smoothing is done.
        :param level: Height of the NGram node.
        """
        nGram.setProbabilityWithPseudoCount(0.0, level)
