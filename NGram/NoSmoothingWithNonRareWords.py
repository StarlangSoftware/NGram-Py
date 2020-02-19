from NGram.NGram import NGram
from NGram.NoSmoothing import NoSmoothing


class NoSmoothingWithNonRareWords(NoSmoothing):

    __dictionary: set
    __probability: float

    def __init__(self, probability: float):
        """
        Constructor of NoSmoothingWithNonRareWords

        PARAMETERS
        ----------
        probability : float
        """
        self.__probability = probability

    def setProbabilities(self, nGram: NGram, level: int):
        """
        Wrapper function to set the N-gram probabilities with no smoothing and replacing unknown words not found in
        nonrare words.

        PARAMETERS
        ----------
        nGram : NGram
            N-Gram for which the probabilities will be set.
        level : int
            Level for which N-Gram probabilities will be set. Probabilities for different levels of the N-gram can be
            set with this function. If level = 1, N-Gram is treated as UniGram, if level = 2, N-Gram is treated as
            Bigram, etc.
        """
        self.__dictionary = nGram.constructDictionaryWithNonRareWords(level, self.__probability)
        nGram.replaceUnknownWords(self.__dictionary)
        super().setProbabilities(nGram, level)
