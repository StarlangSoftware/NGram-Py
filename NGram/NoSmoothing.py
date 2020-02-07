from NGram.NGram import NGram
from NGram.SimpleSmoothing import SimpleSmoothing


class NoSmoothing(SimpleSmoothing):

    def setProbabilities(self, nGram: NGram, level: int):
        nGram.setProbabilityWithPseudoCount(0.0, level)
