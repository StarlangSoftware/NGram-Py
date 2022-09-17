from Sampling.KFoldCrossValidation import KFoldCrossValidation

from NGram.GoodTuringSmoothing import GoodTuringSmoothing
from NGram.NGram import NGram
from NGram.SimpleSmoothing import SimpleSmoothing
from NGram.TrainedSmoothing import TrainedSmoothing
import math


class InterpolatedSmoothing(TrainedSmoothing):

    __lambda1: float
    __lambda2: float
    __simple_smoothing: SimpleSmoothing

    def __init__(self, simpleSmoothing=None):
        """
        Constructor of InterpolatedSmoothing

        PARAMETERS
        ----------
        simpleSmoothing : SimpleSmoothing
            smoothing method.
        """
        if simpleSmoothing is None:
            self.__simple_smoothing = GoodTuringSmoothing()
        else:
            self.__simple_smoothing = simpleSmoothing

    def __learnBestLambda(self,
                          nGrams: list,
                          kFoldCrossValidation: KFoldCrossValidation,
                          lowerBound: float) -> float:
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
        best_previous = -1
        upper_bound = 0.999
        best_lambda = (lowerBound + upper_bound) / 2
        number_of_parts = 5
        test_folds = []
        for i in range(10):
            test_folds.append(kFoldCrossValidation.getTestFold(i))
        while True:
            best_perplexity = 1000000000
            value = lowerBound
            while value <= upper_bound:
                perplexity = 0
                for i in range(10):
                    nGrams[i].setLambda2(value)
                    perplexity += nGrams[i].getPerplexity(test_folds[i])
                if perplexity < best_perplexity:
                    best_perplexity = perplexity
                    best_lambda = value
                value += (upper_bound - lowerBound) / number_of_parts
            lowerBound = self.newLowerBound(best_lambda, lowerBound, upper_bound, number_of_parts)
            upper_bound = self.newUpperBound(best_lambda, lowerBound, upper_bound, number_of_parts)
            if best_previous != -1:
                if math.fabs(best_previous - best_perplexity) / best_perplexity < 0.001:
                    break
            best_previous = best_perplexity
        return best_lambda

    def __learnBestLambdas(self,
                           nGrams: list,
                           kFoldCrossValidation: KFoldCrossValidation,
                           lowerBound1: float,
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
        upper_bound1 = 0.999
        upper_bound2 = 0.999
        best_previous = -1
        best_lambda1 = (lowerBound1 + upper_bound1) / 2
        best_lambda2 = (lowerBound2 + upper_bound2) / 2
        number_of_parts = 5
        test_folds = []
        for i in range(10):
            test_folds.append(kFoldCrossValidation.getTestFold(i))
        while True:
            best_perplexity = 1000000000
            value1 = lowerBound1
            while value1 <= upper_bound1:
                value2 = lowerBound2
                while value2 <= upper_bound2 and value1 + value2 < 1:
                    perplexity = 0
                    for i in range(10):
                        nGrams[i].setLambda3(value1, value2)
                        perplexity += nGrams[i].getPerplexity(test_folds[i])
                    if perplexity < best_perplexity:
                        best_perplexity = perplexity
                        best_lambda1 = value1
                        best_lambda2 = value2
                    value2 += (upper_bound1 - lowerBound1) / number_of_parts
                value1 += (upper_bound1 - lowerBound1) / number_of_parts
            lowerBound1 = self.newLowerBound(best_lambda1, lowerBound1, upper_bound1, number_of_parts)
            upper_bound1 = self.newUpperBound(best_lambda1, lowerBound1, upper_bound1, number_of_parts)
            lowerBound2 = self.newLowerBound(best_lambda2, lowerBound2, upper_bound2, number_of_parts)
            upper_bound2 = self.newUpperBound(best_lambda2, lowerBound2, upper_bound2, number_of_parts)
            if best_previous != -1:
                if math.fabs(best_previous - best_perplexity) / best_perplexity < 0.001:
                    break
            best_previous = best_perplexity
        return best_lambda1, best_lambda2

    def learnParameters(self,
                        corpus: list,
                        N: int):
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
        n_grams = []
        k_fold_cross_validation = KFoldCrossValidation(corpus, K, 0)
        for i in range(K):
            n_grams.append(NGram(N, k_fold_cross_validation.getTrainFold(i)))
            for j in range(2, N + 1):
                n_grams[i].calculateNGramProbabilitiesSimpleLevel(self.__simple_smoothing, j)
            n_grams[i].calculateNGramProbabilitiesSimpleLevel(self.__simple_smoothing, 1)
        if N == 2:
            self.__lambda1 = self.__learnBestLambda(n_grams, k_fold_cross_validation, 0.1)
        elif N == 3:
            (self.__lambda1, self.__lambda2) = self.__learnBestLambdas(n_grams, k_fold_cross_validation, 0.1, 0.1)

    def setProbabilities(self,
                         nGram: NGram,
                         level: int):
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
            nGram.calculateNGramProbabilitiesSimpleLevel(self.__simple_smoothing, j)
        nGram.calculateNGramProbabilitiesSimpleLevel(self.__simple_smoothing, 1)
        if nGram.getN() == 2:
            nGram.setLambda2(self.__lambda1)
        elif nGram.getN() == 3:
            nGram.setLambda3(self.__lambda1, self.__lambda2)
