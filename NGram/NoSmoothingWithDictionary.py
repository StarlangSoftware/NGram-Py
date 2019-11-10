from NGram.NGram import NGram
from NGram.NoSmoothing import NoSmoothing


class NoSmoothingWithDictionary(NoSmoothing):

    __dictionary: set

    """
    Constructor of {@link NoSmoothingWithDictionary}

    PARAMETERS
    ----------
    dictionary : set
        Dictionary to use in smoothing
    """
    def __init__(self, dictionary: set):
        self.__dictionary = dictionary

    """
    Wrapper function to set the N-gram probabilities with no smoothing and replacing unknown words not found in {@link HashSet} the dictionary.

    PARAMETERS
    ----------
    nGram : NGram
        N-Gram for which the probabilities will be set.
    level : int 
        Level for which N-Gram probabilities will be set. Probabilities for different levels of the N-gram can be set 
        with this function. If level = 1, N-Gram is treated as UniGram, if level = 2, N-Gram is treated as Bigram, etc.
    """
    def setProbabilities(self, nGram: NGram, level: int):
        nGram.replaceUnknownWords(self.__dictionary)
        super().setProbabilities(nGram, level)
