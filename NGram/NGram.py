from DataStructure.CounterHashMap import CounterHashMap

from NGram.NGramNode import NGramNode
from NGram.SimpleSmoothing import SimpleSmoothing
from NGram.TrainedSmoothing import TrainedSmoothing
import math


class NGram:

    """
    Constructor of NGram class which takes a list corpus and Integer size of ngram as input.
    It adds all sentences of corpus as ngrams.

    PARAMETERS
    ----------
    N : int
        size of ngram.
    corpus : list
        list of sentences whose ngrams are added.
    """
    def __init__(self, N: int, corpus = None):
        self.N = N
        self.vocabulary = set()
        self.probabilityOfUnseen = []
        self.rootNode = NGramNode(None)
        if corpus is not None:
            for i in range(len(corpus)):
                self.addNGramSentence(corpus[i])

    """
    RETURNS
    -------
    int
        size of ngram.
    """
    def getN(self) -> int:
        return self.N

    """
    Set size of ngram.
    
    PARAMETERS
    ----------
    N : int
        size of ngram
    """
    def setN(self, N: int):
        self.N = N

    """
    Adds given sentence to set the vocabulary and create and add ngrams of the sentence to NGramNode the rootNode

    PARAMETERS
    ----------
    symbols : list
        Sentence whose ngrams are added.
    """
    def addNGramSentence(self, symbols: list):
        for s in symbols:
            self.vocabulary.add(s)
        for j in range(len(symbols) - self.N + 1):
            self.rootNode.addNGram(symbols, j, self.N)

    """
    Adds given array of symbols to set the vocabulary and to NGramNode the rootNode

    PARAMETERS
    ----------
    symbols : list
        ngram added.
    """
    def addNGram(self, symbols: list):
        for s in symbols:
            self.vocabulary.add(s)
        self.rootNode.addNGram(symbols, 0, self.N)

    """
    RETURNS
    -------
    int
        vocabulary size.
    """
    def vocabularySize(self):
        return len(self.vocabulary)

    """
    Sets lambda, interpolation ratio, for bigram and unigram probabilities.
    ie. lambda1 * bigramProbability + (1 - lambda1) * unigramProbability

    PARAMETERS
    ----------
    lambda1 : float
        interpolation ratio for bigram probabilities
    """
    def setLambda2(self, lambda1: float):
        if self.N == 2:
            self.interpolated = True
            self.lambda1 = lambda1

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
    def setLambda3(self, lambda1: float, lambda2: float):
        if self.N == 3:
            self.interpolated = True
            self.lambda1 = lambda1
            self.lambda2 = lambda2

    """
    Calculates NGram probabilities using given corpus and TrainedSmoothing smoothing method.

    PARAMETERS
    ----------
    corpus : list
        corpus for calculating NGram probabilities.
    trainedSmoothing : TrainedSmoothing
        instance of smoothing method for calculating ngram probabilities.
    """
    def calculateNGramProbabilitiesTrained(self, corpus: list, trainedSmoothing: TrainedSmoothing):
        trainedSmoothing.train(corpus, self)

    """
    Calculates NGram probabilities using simple smoothing.

    PARAMETERS
    ----------
    simpleSmoothing : SimpleSmoothing
    """
    def calculateNGramProbabilitiesSimple(self, simpleSmoothing: SimpleSmoothing):
        simpleSmoothing.setProbabilitiesGeneral(self)

    """
    Calculates NGram probabilities given simple smoothing and level.

    PARAMETERS
    ----------
    simpleSmoothing : SimpleSmoothing
    level : int
        Level for which N-Gram probabilities will be set.
    """
    def calculateNGramProbabilitiesSimpleLevel(self, simpleSmoothing: SimpleSmoothing, level: int):
        simpleSmoothing.setProbabilities(self, level)

    """
    Replaces words not in set given dictionary.

    PARAMETERS
    ----------
    dictionary : set
        dictionary of known words.
    """
    def replaceUnknownWords(self, dictionary: set):
        self.rootNode.replaceUnknownWords(dictionary)

    """
    Constructs a dictionary of nonrare words with given N-Gram level and probability threshold.

    PARAMETERS
    ----------
    level : int
        Level for counting words. Counts for different levels of the N-Gram can be set. If level = 1, N-Gram is treated 
        as UniGram, if level = 2, N-Gram is treated as Bigram, etc.
    probability : float
        probability threshold for nonrare words.
        
    RETURNS
    -------
    set
        set of nonrare words.
    """
    def constructDictionaryWithNonRareWords(self, level: int, probability: float) -> set:
        result = set()
        wordCounter = CounterHashMap()
        self.rootNode.countWords(wordCounter, level)
        total = wordCounter.sumOfCounts()
        for symbol in wordCounter.keys():
            if wordCounter[symbol] / total > probability:
                result.add(symbol)
        return result

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
    def getUniGramPerplexity(self, corpus: list) -> float:
        total = 0
        count = 0
        for i in range(len(corpus)):
            for j in range(len(corpus[i])):
                p = self.getProbability(corpus[i][j])
                total -= math.log(p)
                count += 1
        return math.exp(total / count)

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
    def getBiGramPerplexity(self, corpus: list) -> float:
        total = 0
        count = 0
        for i in range(len(corpus)):
            for j in range(len(corpus[i]) - 1):
                p = self.getProbability(corpus[i][j], corpus[i][j + 1])
                total -= math.log(p)
                count += 1
        return math.exp(total / count)

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
    def getTriGramPerplexity(self, corpus: list) -> float:
        total = 0
        count = 0
        for i in range(len(corpus)):
            for j in range(len(corpus[i]) - 2):
                p = self.getProbability(corpus[i][j], corpus[i][j + 1], corpus[i][j + 2])
                total -= math.log(p)
                count += 1
        return math.exp(total / count)

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
    def getPerplexity(self, corpus: list) -> float:
        if self.N == 1:
            return self.getUniGramPerplexity(corpus)
        elif self.N == 2:
            return self.getBiGramPerplexity(corpus)
        elif self.N == 3:
            return self.getTriGramPerplexity(corpus)
        else:
            return 0

    """
    Gets probability of sequence of symbols depending on N in N-Gram. If N is 1, returns unigram probability.
    If N is 2, if interpolated is true, then returns interpolated bigram and unigram probability, otherwise returns only 
    bigram probability.
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
    def getProbability(self, *args) -> float:
        if self.N == 1:
            return self.getUniGramProbability(args[0])
        elif self.N == 2:
            if self.interpolated:
                return self.lambda1 * self.getBiGramProbability(args[0], args[1]) + (1 - self.lambda1) * self.getUniGramProbability(args[1])
            else:
                return self.getBiGramProbability(args[0], args[1])
        elif self.N == 3:
            if self.interpolated:
                return self.lambda1 * self.getTriGramProbability(args[0], args[1], args[2]) + \
                       self.lambda2 * self.getBiGramProbability(args[1], args[2]) + \
                       (1 - self.lambda1 - self.lambda2) * self.getUniGramProbability(args[2])
            else:
                return self.getTriGramProbability(args[0], args[1], args[2])
        else:
            return 0.0

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
    def getUniGramProbability(self, w1) -> float:
        return self.rootNode.getUniGramProbability(w1)

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
    def getBiGramProbability(self, w1, w2) -> float:
        return self.rootNode.getBiGramProbability(w1, w2)

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
    def getTriGramProbability(self, w1, w2, w3) -> float:
        return self.rootNode.getTriGramProbability(w1, w2, w3)

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
    def getCount(self, symbols: list) -> int:
        return self.rootNode.getCountForListItem(symbols, 0)

    """
    Sets probabilities by adding pseudocounts given height and pseudocount.
    
    PARAMETERS
    ----------
    pseudoCount : float
        pseudocount added to all N-Grams.
    height : int
        height for NGram. if height = 1, If level = 1, N-Gram is treated as UniGram, if level = 2, N-Gram is treated as 
        Bigram, etc.
    """
    def setProbabilityWithPseudoCount(self, pseudoCount: float, height: int):
        if pseudoCount != 0:
            vocabularySize = self.vocabularySize() + 1
        else:
            vocabularySize = self.vocabularySize()
        self.rootNode.setProbabilityWithPseudoCount(pseudoCount, height, vocabularySize)
        self.probabilityOfUnseen[height - 1] = 1.0 / vocabularySize

    """
    Find maximum occurrence in given height.
    
    PARAMETERS
    ----------
    height : int
        height for occurrences. If height = 1, N-Gram is treated as UniGram, if height = 2, N-Gram is treated as Bigram, 
        etc.
        
    RETURNS
    -------
    int
        maximum occurrence in given height.
    """
    def maximumOccurence(self, height: int) -> int:
        return self.rootNode.maximumOccurence(height)

    """
    Update counts of counts of N-Grams with given counts of counts and given height.
    
    PARAMETERS
    ----------
    countsOfCounts : list
        updated counts of counts.
    height : int
        height for NGram. If height = 1, N-Gram is treated as UniGram, if height = 2, N-Gram is treated as Bigram, etc.
    """
    def updateCountsOfCounts(self, countsOfCounts: list, height: int):
        self.rootNode.updateCountsOfCounts(countsOfCounts, height)

    """
    Calculates counts of counts of NGrams.
    
    PARAMETERS
    ----------
    height : int  
        height for NGram. If height = 1, N-Gram is treated as UniGram, if height = 2, N-Gram is treated as Bigram, etc.

    RETURNS
    -------
    list
        counts of counts of NGrams.
    """
    def calculateCountsOfCounts(self, height: int) -> list:
        maxCount = self.maximumOccurence(height)
        countsOfCounts = [0] * (maxCount + 2)
        self.updateCountsOfCounts(countsOfCounts, height)
        return countsOfCounts

    """
    Sets probability with given counts of counts and pZero.
    
    PARAMETERS
    ----------
    countsOfCounts : list
        counts of counts of NGrams.
    height : int 
        height for NGram. If height = 1, N-Gram is treated as UniGram, if height = 2, N-Gram is treated as Bigram, etc.
    pZero : float
        probability of zero.
    """
    def setAdjustedProbability(self, countsOfCounts: list, height: int, pZero: float):
        self.rootNode.setAdjustedProbability(countsOfCounts, height, self.vocabularySize() + 1, pZero)
        self.probabilityOfUnseen[height - 1] = 1.0 / (self.vocabularySize() + 1)
