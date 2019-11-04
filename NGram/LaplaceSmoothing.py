from NGram.NGram import NGram
from NGram.SimpleSmoothing import SimpleSmoothing


class LaplaceSmoothing(SimpleSmoothing):

    def __init__(self, delta = 1.0):
        self.delta = delta

    """
    Wrapper function to set the N-gram probabilities with laplace smoothing.

    PARAMETERS
    ----------
    nGram : NGram
        N-Gram for which the probabilities will be set.
    level : int
        height for NGram. if level = 1, If level = 1, N-Gram is treated as UniGram, if level = 2, N-Gram is treated as 
        Bigram, etc.
    """
    def setProbabilities(self, nGram: NGram, level: int):
        nGram.setProbabilityWithPseudoCount(self.delta, level)
