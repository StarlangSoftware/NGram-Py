from __future__ import annotations
from DataStructure.CounterHashMap import CounterHashMap

from NGram.NGramNode import NGramNode
from NGram.SimpleSmoothing import SimpleSmoothing
from NGram.TrainedSmoothing import TrainedSmoothing
from NGram.MultipleFile import MultipleFile
import math


class NGram:

    rootNode: NGramNode
    __N: int
    __lambda1: float
    __lambda2: float
    __interpolated: bool
    __vocabulary: set
    __probability_of_unseen: list

    def __init__(self,
                 NorFileName,
                 corpus=None):
        """
        Constructor of NGram class which takes a list corpus and Integer size of ngram as input.
        It adds all sentences of corpus as ngrams.

        PARAMETERS
        ----------
        NorFileName
            size of ngram.
            fileName
        corpus : list
            list of sentences whose ngrams are added.
        """
        if isinstance(NorFileName, int):
            self.__N = NorFileName
            self.__vocabulary = set()
            self.__probability_of_unseen = self.__N * [0.0]
            self.__lambda1 = 0.0
            self.__lambda2 = 0.0
            self.__interpolated = False
            self.rootNode = NGramNode(None)
            if corpus is not None:
                for i in range(len(corpus)):
                    self.addNGramSentence(corpus[i])
        else:
            inputFile = open(NorFileName, mode="r", encoding="utf-8")
            line = inputFile.readline().strip()
            items = line.split()
            self.__N = int(items[0])
            self.__lambda1 = float(items[1])
            self.__lambda2 = float(items[2])
            self.__probability_of_unseen = self.__N * [0.0]
            self.__interpolated = False
            line = inputFile.readline().strip()
            items = line.split()
            for i in range(len(items)):
                self.__probability_of_unseen[i] = float(items[i])
            self.__vocabulary = set()
            vocabulary_size = int(inputFile.readline().strip())
            for i in range(vocabulary_size):
                self.__vocabulary.add(inputFile.readline().strip())
            self.rootNode = NGramNode(True, inputFile)
            inputFile.close()

    def initWithMultipleFile(self, *args):
        multiple_file = MultipleFile(list(args))
        line = multiple_file.readLine().strip()
        items = line.split()
        self.__N = int(items[0])
        self.__lambda1 = float(items[1])
        self.__lambda2 = float(items[2])
        self.__probability_of_unseen = self.__N * [0.0]
        self.__interpolated = False
        line = multiple_file.readLine().strip()
        items = line.split()
        for i in range(len(items)):
            self.__probability_of_unseen[i] = float(items[i])
        self.__vocabulary = set()
        vocabulary_size = int(multiple_file.readLine().strip())
        for i in range(vocabulary_size):
            self.__vocabulary.add(multiple_file.readLine().strip())
        self.rootNode = NGramNode(True, multiple_file)

    def merge(self, toBeMerged: NGram):
        """
        Merges current NGram with the given NGram. If N of the two NGram's are not same, it does not
        merge. Merges first the vocabulary, then the NGram trees.
        :param toBeMerged: NGram to be merged with.
        """
        if self.__N != toBeMerged.getN():
            return
        self.__vocabulary.update(toBeMerged.__vocabulary)
        self.rootNode.merge(toBeMerged.rootNode)

    def getN(self) -> int:
        """
        RETURNS
        -------
        int
            size of ngram.
        """
        return self.__N

    def setN(self, N: int):
        """
        Set size of ngram.

        PARAMETERS
        ----------
        N : int
            size of ngram
        """
        self.__N = N

    def addNGramSentence(self,
                         symbols: list,
                         sentenceCount: int = 1):
        """
        Adds given sentence to set the vocabulary and create and add ngrams of the sentence to NGramNode the rootNode

        PARAMETERS
        ----------
        symbols : list
            Sentence whose ngrams are added.
        sentenceCount : int
            Number of times this sentence is added.
        """
        for s in symbols:
            self.__vocabulary.add(s)
        for j in range(len(symbols) - self.__N + 1):
            self.rootNode.addNGram(symbols, j, self.__N, sentenceCount)

    def addNGram(self, symbols: list):
        """
        Adds given array of symbols to set the vocabulary and to NGramNode the rootNode

        PARAMETERS
        ----------
        symbols : list
            ngram added.
        """
        for s in symbols:
            self.__vocabulary.add(s)
        self.rootNode.addNGram(symbols, 0, self.__N)

    def vocabularySize(self) -> int:
        """
        RETURNS
        -------
        int
            vocabulary size.
        """
        return len(self.__vocabulary)

    def setLambda2(self, lambda1: float):
        """
        Sets lambda, interpolation ratio, for bigram and unigram probabilities.
        ie. lambda1 * bigramProbability + (1 - lambda1) * unigramProbability

        PARAMETERS
        ----------
        lambda1 : float
            interpolation ratio for bigram probabilities
        """
        if self.__N == 2:
            self.__interpolated = True
            self.__lambda1 = lambda1

    def setLambda3(self, lambda1: float, lambda2: float):
        """
        Sets lambdas, interpolation ratios, for trigram, bigram and unigram probabilities.
        ie. lambda1 * trigramProbability + lambda2 * bigramProbability  + (1 - lambda1 - lambda2) * unigramProbability

        PARAMETERS
        ----------
        lambda1 : float
            interpolation ratio for trigram probabilities
        lambda2 : float
            interpolation ratio for bigram probabilities
        """
        if self.__N == 3:
            self.__interpolated = True
            self.__lambda1 = lambda1
            self.__lambda2 = lambda2

    def calculateNGramProbabilitiesTrained(self,
                                           corpus: list,
                                           trainedSmoothing: TrainedSmoothing):
        """
        Calculates NGram probabilities using given corpus and TrainedSmoothing smoothing method.

        PARAMETERS
        ----------
        corpus : list
            corpus for calculating NGram probabilities.
        trainedSmoothing : TrainedSmoothing
            instance of smoothing method for calculating ngram probabilities.
        """
        trainedSmoothing.train(corpus, self)

    def calculateNGramProbabilitiesSimple(self, simpleSmoothing: SimpleSmoothing):
        """
        Calculates NGram probabilities using simple smoothing.

        PARAMETERS
        ----------
        simpleSmoothing : SimpleSmoothing
        """
        simpleSmoothing.setProbabilitiesGeneral(self)

    def calculateNGramProbabilitiesSimpleLevel(self,
                                               simpleSmoothing: SimpleSmoothing,
                                               level: int):
        """
        Calculates NGram probabilities given simple smoothing and level.

        PARAMETERS
        ----------
        simpleSmoothing : SimpleSmoothing
        level : int
            Level for which N-Gram probabilities will be set.
        """
        simpleSmoothing.setProbabilities(self, level)

    def replaceUnknownWords(self, dictionary: set):
        """
        Replaces words not in set given dictionary.

        PARAMETERS
        ----------
        dictionary : set
            dictionary of known words.
        """
        self.rootNode.replaceUnknownWords(dictionary)

    def constructDictionaryWithNonRareWords(self,
                                            level: int,
                                            probability: float) -> set:
        """
        Constructs a dictionary of nonrare words with given N-Gram level and probability threshold.

        PARAMETERS
        ----------
        level : int
            Level for counting words. Counts for different levels of the N-Gram can be set. If level = 1, N-Gram is
            treated as UniGram, if level = 2, N-Gram is treated as Bigram, etc.
        probability : float
            probability threshold for nonrare words.

        RETURNS
        -------
        set
            set of nonrare words.
        """
        result = set()
        word_counter = CounterHashMap()
        self.rootNode.countWords(word_counter, level)
        total = word_counter.sumOfCounts()
        for symbol in word_counter.keys():
            if word_counter[symbol] / total > probability:
                result.add(symbol)
        return result

    def __getUniGramPerplexity(self, corpus: list) -> float:
        """
        Calculates unigram perplexity of given corpus. First sums negative log likelihoods of all unigrams in corpus.
        Then returns exp of average negative log likelihood.

        PARAMETERS
        ----------
        corpus : list
            corpus whose unigram perplexity is calculated.

        RETURNS
        -------
        float
            unigram perplexity of corpus.
        """
        total = 0
        count = 0
        for i in range(len(corpus)):
            for j in range(len(corpus[i])):
                p = self.getProbability(corpus[i][j])
                total -= math.log(p)
                count += 1
        return math.exp(total / count)

    def __getBiGramPerplexity(self, corpus: list) -> float:
        """
        Calculates bigram perplexity of given corpus. First sums negative log likelihoods of all bigrams in corpus.
        Then returns exp of average negative log likelihood.

        PARAMETERS
        ----------
        corpus : list
            corpus whose bigram perplexity is calculated.

        RETURNS
        -------
        float
            bigram perplexity of corpus.
        """
        total = 0
        count = 0
        for i in range(len(corpus)):
            for j in range(len(corpus[i]) - 1):
                p = self.getProbability(corpus[i][j],
                                        corpus[i][j + 1])
                total -= math.log(p)
                count += 1
        return math.exp(total / count)

    def __getTriGramPerplexity(self, corpus: list) -> float:
        """
        Calculates trigram perplexity of given corpus. First sums negative log likelihoods of all trigrams in corpus.
        Then returns exp of average negative log likelihood.

        PARAMETERS
        ----------
        corpus : list
            corpus whose trigram perplexity is calculated.

        RETURNS
        -------
        float
            trigram perplexity of corpus.
        """
        total = 0
        count = 0
        for i in range(len(corpus)):
            for j in range(len(corpus[i]) - 2):
                p = self.getProbability(corpus[i][j],
                                        corpus[i][j + 1],
                                        corpus[i][j + 2])
                total -= math.log(p)
                count += 1
        return math.exp(total / count)

    def getPerplexity(self, corpus: list) -> float:
        """
        Calculates the perplexity of given corpus depending on N-Gram model (unigram, bigram, trigram, etc.)

        PARAMETERS
        ----------
        corpus : list
            corpus whose perplexity is calculated.

        RETURNS
        -------
        float
            perplexity of given corpus
        """
        if self.__N == 1:
            return self.__getUniGramPerplexity(corpus)
        elif self.__N == 2:
            return self.__getBiGramPerplexity(corpus)
        elif self.__N == 3:
            return self.__getTriGramPerplexity(corpus)
        else:
            return 0

    def getProbability(self, *args) -> float:
        """
        Gets probability of sequence of symbols depending on N in N-Gram. If N is 1, returns unigram probability.
        If N is 2, if interpolated is true, then returns interpolated bigram and unigram probability, otherwise returns
        only bigram probability.
        If N is 3, if interpolated is true, then returns interpolated trigram, bigram and unigram probability, otherwise
        returns only trigram probability.

        PARAMETERS
        ----------
        args
            symbols sequence of symbol.

        RETURNS
        -------
        float
            probability of given sequence.
        """
        if self.__N == 1:
            return self.__getUniGramProbability(args[0])
        elif self.__N == 2:
            if len(args) == 1:
                return self.__getUniGramProbability(args[0])
            if self.__interpolated:
                return self.__lambda1 * self.__getBiGramProbability(args[0], args[1]) + (1 - self.__lambda1) \
                       * self.__getUniGramProbability(args[1])
            else:
                return self.__getBiGramProbability(args[0], args[1])
        elif self.__N == 3:
            if len(args) == 1:
                return self.__getUniGramProbability(args[0])
            elif len(args) == 2:
                return self.__getBiGramProbability(args[0], args[1])
            if self.__interpolated:
                return self.__lambda1 * self.__getTriGramProbability(args[0], args[1], args[2]) + \
                       self.__lambda2 * self.__getBiGramProbability(args[1], args[2]) + \
                       (1 - self.__lambda1 - self.__lambda2) * self.__getUniGramProbability(args[2])
            else:
                return self.__getTriGramProbability(args[0], args[1], args[2])
        else:
            return 0.0

    def __getUniGramProbability(self, w1) -> float:
        """
        Gets unigram probability of given symbol.

        PARAMETERS
        ----------
        w1
            a unigram symbol.

        RETURNS
        -------
        float
            probability of given unigram.
        """
        return self.rootNode.getUniGramProbability(w1)

    def __getBiGramProbability(self, w1, w2) -> float:
        """
        Gets bigram probability of given symbols.

        PARAMETERS
        ----------
        w1
            first gram of bigram
        w2
            second gram of bigram

        RETURNS
        -------
        float
            probability of bigram formed by w1 and w2.
        """
        probability = self.rootNode.getBiGramProbability(w1, w2)
        if probability is not None:
            return probability
        else:
            return self.__probability_of_unseen[1]

    def __getTriGramProbability(self, w1, w2, w3) -> float:
        """
        Gets trigram probability of given symbols.

        PARAMETERS
        ----------
        w1
            first gram of trigram
        w2
            second gram of trigram
        w3
            third gram of trigram

        RETURNS
        -------
        float
            probability of trigram formed by w1, w2, w3.
        """
        probability = self.rootNode.getTriGramProbability(w1, w2, w3)
        if probability is not None:
            return probability
        else:
            return self.__probability_of_unseen[2]

    def getCount(self, symbols: list) -> int:
        """
        Gets count of given sequence of symbol.

        PARAMETERS
        ----------
        symbols : list
            sequence of symbol.

        RETURNS
        -------
        int
            count of symbols.
        """
        return self.rootNode.getCountForListItem(symbols, 0)

    def setProbabilityWithPseudoCount(self,
                                      pseudoCount: float,
                                      height: int):
        """
        Sets probabilities by adding pseudocounts given height and pseudocount.

        PARAMETERS
        ----------
        pseudoCount : float
            pseudocount added to all N-Grams.
        height : int
            height for NGram. if height = 1, If level = 1, N-Gram is treated as UniGram, if level = 2, N-Gram is treated
            as Bigram, etc.
        """
        if pseudoCount != 0:
            vocabulary_size = self.vocabularySize() + 1
        else:
            vocabulary_size = self.vocabularySize()
        self.rootNode.setProbabilityWithPseudoCount(pseudoCount, height, vocabulary_size)
        if pseudoCount != 0:
            self.__probability_of_unseen[height - 1] = 1.0 / vocabulary_size
        else:
            self.__probability_of_unseen[height - 1] = 0.0

    def __maximumOccurence(self, height: int) -> int:
        """
        Find maximum occurrence in given height.

        PARAMETERS
        ----------
        height : int
            height for occurrences. If height = 1, N-Gram is treated as UniGram, if height = 2, N-Gram is treated as
            Bigram,
            etc.

        RETURNS
        -------
        int
            maximum occurrence in given height.
        """
        return self.rootNode.maximumOccurence(height)

    def __updateCountsOfCounts(self,
                               countsOfCounts: list,
                               height: int):
        """
        Update counts of counts of N-Grams with given counts of counts and given height.

        PARAMETERS
        ----------
        countsOfCounts : list
            updated counts of counts.
        height : int
            height for NGram. If height = 1, N-Gram is treated as UniGram, if height = 2, N-Gram is treated as Bigram,
            etc.
        """
        self.rootNode.updateCountsOfCounts(countsOfCounts, height)

    def calculateCountsOfCounts(self, height: int) -> list:
        """
        Calculates counts of counts of NGrams.

        PARAMETERS
        ----------
        height : int
            height for NGram. If height = 1, N-Gram is treated as UniGram, if height = 2, N-Gram is treated as Bigram,
            etc.

        RETURNS
        -------
        list
            counts of counts of NGrams.
        """
        max_count = self.__maximumOccurence(height)
        counts_of_counts = [0] * (max_count + 2)
        self.__updateCountsOfCounts(counts_of_counts, height)
        return counts_of_counts

    def setAdjustedProbability(self,
                               countsOfCounts: list,
                               height: int,
                               pZero: float):
        """
        Sets probability with given counts of counts and pZero.

        PARAMETERS
        ----------
        countsOfCounts : list
            counts of counts of NGrams.
        height : int
            height for NGram. If height = 1, N-Gram is treated as UniGram, if height = 2, N-Gram is treated as Bigram,
            etc.
        pZero : float
            probability of zero.
        """
        self.rootNode.setAdjustedProbability(countsOfCounts, height, self.vocabularySize() + 1, pZero)
        self.__probability_of_unseen[height - 1] = 1.0 / (self.vocabularySize() + 1)

    def prune(self, threshold: float):
        """
        Prunes NGram according to the given threshold. All nodes having a probability less than the threshold will be
        pruned.
        :param threshold: Probability threshold used for pruning.
        """
        if 0.0 < threshold <= 1.0:
            self.rootNode.prune(threshold, self.__N - 1)

    def saveAsText(self, fileName: str):
        """
        Save this NGram to a text file.

        PARAMETERS
        ----------
        fileName : str
            String name of file where NGram is saved.
        """
        output_file = open(fileName, mode="w", encoding="utf8")
        output_file.write(self.__N.__str__() + " " + self.__lambda1.__str__() + " " + self.__lambda2.__str__() + "\n")
        for p in self.__probability_of_unseen:
            output_file.write(p.__str__() + " ")
        output_file.write("\n")
        output_file.write(self.vocabularySize().__str__() + "\n")
        for symbol in self.__vocabulary:
            output_file.write(symbol.__str__() + "\n")
        self.rootNode.saveAsText(True, output_file, 0)
        output_file.close()
