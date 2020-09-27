from Sampling.KFoldCrossValidation import KFoldCrossValidation

from NGram.GoodTuringSmoothing import GoodTuringSmoothing
from NGram.NGram import NGram
from NGram.SimpleSmoothing import SimpleSmoothing
from NGram.TrainedSmoothing import TrainedSmoothing
import math


class InterpolatedSmoothing(TrainedSmoothing):

    __lambda1: float
    __lambda2: float
    __simpleSmoothing: SimpleSmoothing

    def __init__(self, simpleSmoothing=None):
        """
        Constructor of InterpolatedSmoothing

        PARAMETERS
        ----------
        simpleSmoothing : SimpleSmoothing
            smoothing method.
        """
        if simpleSmoothing is None:
            self.__simpleSmoothing = GoodTuringSmoothing()
        else:
            self.__simpleSmoothing = simpleSmoothing

    def __learnBestLambda(self, nGrams: list, kFoldCrossValidation: KFoldCrossValidation, lowerBound: float) -> float:
        """
        The algorithm tries to optimize the best lambda for a given corpus. The algorithm uses perplexity on the
        validation set as the optimization criterion.

        PARAMETERS
        ----------
        nGrams : list
            10 N-Grams learned for different folds of the corpus. nGrams[i] is the N-Gram trained with i'th train fold
            of the corpus.
        kFoldCrossValidation : KFoldCrossvalidation
            Cross-validation data used in training and testing the N-grams.
        lowerBound : float
            Initial lower bound for optimizing the best lambda.

        RETURNS
        -------
        float
            Best lambda optimized with k-fold crossvalidation.
        """
        bestPrevious = -1
        upperBound = 0.999
        bestLambda = (lowerBound + upperBound) / 2
        numberOfParts = 5
        testFolds = []
        for i in range(10):
            testFolds.append(kFoldCrossValidation.getTestFold(i))
        while True:
            bestPerplexity = 1000000000
            value = lowerBound
            while value <= upperBound:
                perplexity = 0
                for i in range(10):
                    nGrams[i].setLambda2(value)
                    perplexity += nGrams[i].getPerplexity(testFolds[i])
                if perplexity < bestPerplexity:
                    bestPerplexity = perplexity
                    bestLambda = value
                value += (upperBound - lowerBound) / numberOfParts
            lowerBound = self.newLowerBound(bestLambda, lowerBound, upperBound, numberOfParts)
            upperBound = self.newUpperBound(bestLambda, lowerBound, upperBound, numberOfParts)
            if bestPrevious != -1:
                if math.fabs(bestPrevious - bestPerplexity) / bestPerplexity < 0.001:
                    break
            bestPrevious = bestPerplexity
        return bestLambda

    def __learnBestLambdas(self, nGrams: list, kFoldCrossValidation: KFoldCrossValidation, lowerBound1: float,
                           lowerBound2: float) -> tuple:
        """
        The algorithm tries to optimize the best lambdas (lambda1, lambda2) for a given corpus. The algorithm uses
        perplexity on the validation set as the optimization criterion.

        PARAMETERS
        ----------
        nGrams : list
            10 N-Grams learned for different folds of the corpus. nGrams[i] is the N-Gram trained with i'th train fold
            of the corpus.
        kFoldCrossValidation : KFoldCrossValidation
            Cross-validation data used in training and testing the N-grams.
        lowerBound1 : float
            Initial lower bound for optimizing the best lambda1.
        lowerBound2 : float
            Initial lower bound for optimizing the best lambda2.

        RETURNS
        -------
        tuple
            bestLambda1 and bestLambda2
        """
        upperBound1 = 0.999
        upperBound2 = 0.999
        bestPrevious = -1
        bestLambda1 = (lowerBound1 + upperBound1) / 2
        bestLambda2 = (lowerBound2 + upperBound2) / 2
        numberOfParts = 5
        testFolds = []
        for i in range(10):
            testFolds.append(kFoldCrossValidation.getTestFold(i))
        while True:
            bestPerplexity = 1000000000
            value1 = lowerBound1
            while value1 <= upperBound1:
                value2 = lowerBound2
                while value2 <= upperBound2 and value1 + value2 < 1:
                    perplexity = 0
                    for i in range(10):
                        nGrams[i].setLambda3(value1, value2)
                        perplexity += nGrams[i].getPerplexity(testFolds[i])
                    if perplexity < bestPerplexity:
                        bestPerplexity = perplexity
                        bestLambda1 = value1
                        bestLambda2 = value2
                    value2 += (upperBound1 - lowerBound1) / numberOfParts
                value1 += (upperBound1 - lowerBound1) / numberOfParts
            lowerBound1 = self.newLowerBound(bestLambda1, lowerBound1, upperBound1, numberOfParts)
            upperBound1 = self.newUpperBound(bestLambda1, lowerBound1, upperBound1, numberOfParts)
            lowerBound2 = self.newLowerBound(bestLambda2, lowerBound2, upperBound2, numberOfParts)
            upperBound2 = self.newUpperBound(bestLambda2, lowerBound2, upperBound2, numberOfParts)
            if bestPrevious != -1:
                if math.fabs(bestPrevious - bestPerplexity) / bestPerplexity < 0.001:
                    break
            bestPrevious = bestPerplexity
        return bestLambda1, bestLambda2

    def learnParameters(self, corpus: list, N: int):
        """
        Wrapper function to learn the parameters (lambda1 and lambda2) in interpolated smoothing. The function first
        creates K NGrams with the train folds of the corpus. Then optimizes lambdas with respect to the test folds of
        the corpus depending on given N.

        PARAMETERS
        ----------
        corpus : list
            Train corpus used to optimize lambda parameters
        N : int
            N in N-Gram.
        """
        if N <= 1:
            return
        K = 10
        nGrams = []
        kFoldCrossValidation = KFoldCrossValidation(corpus, K, 0)
        for i in range(K):
            nGrams.append(NGram(N, kFoldCrossValidation.getTrainFold(i)))
            for j in range(2, N + 1):
                nGrams[i].calculateNGramProbabilitiesSimpleLevel(self.__simpleSmoothing, j)
            nGrams[i].calculateNGramProbabilitiesSimpleLevel(self.__simpleSmoothing, 1)
        if N == 2:
            self.__lambda1 = self.__learnBestLambda(nGrams, kFoldCrossValidation, 0.1)
        elif N == 3:
            (self.__lambda1, self.__lambda2) = self.__learnBestLambdas(nGrams, kFoldCrossValidation, 0.1, 0.1)

    def setProbabilities(self, nGram: NGram, level: int):
        """
        Wrapper function to set the N-gram probabilities with interpolated smoothing.

        PARAMETERS
        ----------
        nGram : NGram
            N-Gram for which the probabilities will be set.
        level : int
            Level for which N-Gram probabilities will be set. Probabilities for different levels of the N-gram can be
            set with this function. If level = 1, N-Gram is treated as UniGram, if level = 2, N-Gram is treated as
            Bigram, etc.
        """
        for j in range(2, nGram.getN() + 1):
            nGram.calculateNGramProbabilitiesSimpleLevel(self.__simpleSmoothing, j)
        nGram.calculateNGramProbabilitiesSimpleLevel(self.__simpleSmoothing, 1)
        if nGram.getN() == 2:
            nGram.setLambda2(self.__lambda1)
        elif nGram.getN() == 3:
            nGram.setLambda3(self.__lambda1, self.__lambda2)
