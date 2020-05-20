from Math.Matrix import Matrix
from Math.Vector import Vector

from NGram.NGram import NGram
from NGram.SimpleSmoothing import SimpleSmoothing
import math


class GoodTuringSmoothing(SimpleSmoothing):

    def __linearRegressionOnCountsOfCounts(self, countsOfCounts: list) -> list:
        """
        Given counts of counts, this function will calculate the estimated counts of counts c$^*$ with
        Good-Turing smoothing. First, the algorithm filters the non-zero counts from counts of counts array and constructs
        c and r arrays. Then it constructs Z_n array with Z_n = (2C_n / (r_{n+1} - r_{n-1})). The algorithm then uses
        simple linear regression on Z_n values to estimate w_1 and w_0, where log(N[i]) = w_1log(i) + w_0

        PARAMETERS
        ----------
        countsOfCounts : list
            Counts of counts. countsOfCounts[1] is the number of words occurred once in the corpus. countsOfCounts[i] is
            the number of words occurred i times in the corpus.

        RETURNS
        ------
        list
            Estimated counts of counts array. N[1] is the estimated count for out of vocabulary words.
        """
        N = [0.0] * len(countsOfCounts)
        r = []
        c = []
        for i in range(1, len(countsOfCounts)):
            if countsOfCounts[i] != 0:
                r.append(i)
                c.append(countsOfCounts[i])
        A = Matrix(2, 2)
        y = Vector(2, 0)
        for i in range(len(r)):
            xt = math.log(r[i])
            if i == 0:
                rt = math.log(c[i])
            else:
                if i == len(r) - 1:
                    rt = math.log((1.0 * c[i]) / (r[i] - r[i - 1]))
                else:
                    rt = math.log((2.0 * c[i]) / (r[i + 1] - r[i - 1]))
            A.addValue(0, 0, 1.0)
            A.addValue(0, 1, xt)
            A.addValue(1, 0, xt)
            A.addValue(1, 1, xt * xt)
            y.addValue(0, rt)
            y.addValue(1, rt * xt)
        A.inverse()
        w = A.multiplyWithVectorFromRight(y)
        w0 = w.getValue(0)
        w1 = w.getValue(1)
        for i in range(1, len(countsOfCounts)):
            N[i] = math.exp(math.log(i) * w1 + w0)
        return N

    def setProbabilities(self, nGram: NGram, level: int):
        """
        Wrapper function to set the N-gram probabilities with Good-Turing smoothing. N[1] / sum_{i=1}^infinity N_i is
        the out of vocabulary probability.

        PARAMETERS
        ----------
        nGram : NGram
            N-Gram for which the probabilities will be set.
        level : int
            Level for which N-Gram probabilities will be set. Probabilities for different levels of the N-gram can be
            set with this function. If level = 1, N-Gram is treated as UniGram, if level = 2, N-Gram is treated as
            Bigram, etc.
        """
        countsOfCounts = nGram.calculateCountsOfCounts(level)
        N = self.__linearRegressionOnCountsOfCounts(countsOfCounts)
        total = 0.0
        for r in range(1, len(countsOfCounts)):
            total += countsOfCounts[r] * r
        nGram.setAdjustedProbability(N, level, N[1] / total)
