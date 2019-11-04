from NGram.NGram import NGram
from NGram.NoSmoothing import NoSmoothing


class NoSmoothingWithNonRareWords(NoSmoothing):

    """
    Constructor of NoSmoothingWithNonRareWords

    PARAMETERS
    ----------
    probability : float
    """
    def __init__(self, probability: float):
        self.probability = probability

    """
    Wrapper function to set the N-gram probabilities with no smoothing and replacing unknown words not found in nonrare words.

    PARAMETERS
    ----------
    nGram : NGram
        N-Gram for which the probabilities will be set.
    level : int
        Level for which N-Gram probabilities will be set. Probabilities for different levels of the N-gram can be set
         with this function. If level = 1, N-Gram is treated as UniGram, if level = 2, N-Gram is treated as Bigram, etc.
    """
    def setProbabilities(self, nGram: NGram, level: int):
        self.dictionary = nGram.constructDictionaryWithNonRareWords(level, self.probability)
        nGram.replaceUnknownWords(self.dictionary)
        super().setProbabilities(nGram, level)
