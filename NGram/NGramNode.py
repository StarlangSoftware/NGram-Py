from __future__ import annotations
from DataStructure.CounterHashMap import CounterHashMap
import random


class NGramNode(object):

    __children: dict
    __symbol: object
    __count: int
    __probability: float
    __probabilityOfUnseen: float
    __unknown: NGramNode

    """
    Constructor of NGramNode

    PARAMETERS
    ----------
    symbol
        symbol to be kept in this node.
    """
    def __init__(self, symbol):
        self.__symbol = symbol
        self.__count = 0
        self.__children = {}

    """
    Gets count of this node.
    
    RETURNS
    -------
    int
        count of this node.
    """
    def getCount(self) -> int:
        return self.__count

    """
    Gets the size of children of this node.
    
    RETURNS
    -------
    int
        size of children of NGramNode this node.
    """
    def size(self) -> int:
        return len(self.__children)

    """
    Finds maximum occurrence. If height is 0, returns the count of this node.
    Otherwise, traverses this nodes' children recursively and returns maximum occurrence.

    PARAMETERS
    ----------
    height : int
        height for NGram.
        
    RETURNS
    -------
    int
        maximum occurrence.
    """
    def maximumOccurence(self, height: int) -> int:
        maxValue = 0
        if height == 0:
            return self.__count
        else:
            for child in self.__children.values():
                current = child.maximumOccurence(height - 1)
                if current > maxValue:
                    maxValue = current
            return maxValue

    """
    RETURNS
    -------
    float
        sum of counts of children nodes.
    """
    def childSum(self) -> float:
        total = 0
        for child in self.__children.values():
            total += child.count
        if self.__unknown is not None:
            total += self.__unknown.__count
        return total

    """
    Traverses nodes and updates counts of counts for each node.
    
    PARAMETERS
    ----------
    countsOfCounts : list
        counts of counts of NGrams.
    height : int
        height for NGram. if height = 1, If level = 1, N-Gram is treated as UniGram, if level = 2, N-Gram is treated as 
        Bigram, etc.
    """
    def updateCountsOfCounts(self, countsOfCounts: list, height: int):
        if height == 0:
            countsOfCounts[self.__count] = countsOfCounts[self.__count] + 1
        else:
            for child in self.__children.values():
                child.updateCountsOfCounts(countsOfCounts, height - 1)

    """
    Sets probabilities by traversing nodes and adding pseudocount for each NGram.

    PARAMETERS
    ----------
    pseudoCount : int
        pseudocount added to each NGram.
    height : int
        height for NGram. if height = 1, If level = 1, N-Gram is treated as UniGram, if level = 2, N-Gram is treated as 
        Bigram, etc.
    vocabularySize : float
        size of vocabulary
    """
    def setProbabilityWithPseudoCount(self, pseudoCount: float, height: int, vocabularySize: float):
        if height == 1:
            total = self.childSum() + pseudoCount * vocabularySize
            for child in self.__children.values():
                child.probability = (child.count + pseudoCount) / total
            if self.__unknown is not None:
                self.__unknown.__probability = (self.__unknown.__count + pseudoCount) / total
            self.__probabilityOfUnseen = pseudoCount / total
        else:
            for child in self.__children.values():
                child.setProbabilityWithPseudoCount(pseudoCount, height - 1, vocabularySize)

    """
    Sets adjusted probabilities with counts of counts of NGrams.
    For count < 5, count is considered as ((r + 1) * N[r + 1]) / N[r]), otherwise, count is considered as it is.
    Sum of children counts are computed. Then, probability of a child node is (1 - pZero) * (r / sum) if r > 5
    otherwise, r is replaced with ((r + 1) * N[r + 1]) / N[r]) and calculated the same.

    PARAMETERS
    ----------
    N : list
        counts of counts of NGrams.
    height : int
        height for NGram. if height = 1, If level = 1, N-Gram is treated as UniGram, if level = 2, N-Gram is treated as 
        Bigram, etc.
    vocabularySize : float
        size of vocabulary.
    pZero : float
        probability of zero.
    """
    def setAdjustedProbability(self, N: list, height: int, vocabularySize: float, pZero: float):
        if height == 1:
            total = 0
            for child in self.__children.values():
                r = child.count
                if r <= 5:
                    newR = ((r + 1) * N[r + 1]) / N[r]
                    total += newR
                else:
                    total += r
            for child in self.__children.values():
                r = child.count
                if r <= 5:
                    newR = ((r + 1) * N[r + 1]) / N[r]
                    child.probability = (1 - pZero) * (newR / total)
                else:
                    child.probability = (1 - pZero) * (r / total)
            self.__probabilityOfUnseen = pZero / (vocabularySize - len(self.__children))
        else:
            for child in self.__children.values():
                child.setAdjustedProbability(N, height - 1, vocabularySize, pZero)

    """
    Adds NGram given as array of symbols to the node as a child.

    PARAMETERS
    ----------
    s : list
        array of symbols
    index : int
        start index of NGram
    height : int
        height for NGram. if height = 1, If level = 1, N-Gram is treated as UniGram, if level = 2, N-Gram is treated as 
        Bigram, etc.
    """
    def addNGram(self, s: list, index: int, height: int):
        if height == 0:
            return
        symbol = s[index]
        if symbol in self.__children:
            child = self.__children[symbol]
        else:
            child = NGramNode(symbol)
            self.__children[symbol] = child
        child.__count += 1
        child.addNGram(s, index + 1, height - 1)

    """
    Gets unigram probability of given symbol.

    PARAMETERS
    ----------
    w1
        unigram.
        
    RETURNS
    -------
    float
        unigram probability of given symbol.
    """
    def getUniGramProbability(self, w1) -> float:
        if w1 in self.__children:
            return self.__children[w1].probability
        elif self.__unknown is not None:
            return self.__unknown.__probability
        else:
            return self.__probabilityOfUnseen

    """
    Gets bigram probability of given symbols w1 and w2

    PARAMETERS
    ----------
    w1 
        first gram of bigram.
    w2 
        second gram of bigram.
        
    RETURNS
    -------
    float
        probability of given bigram
    """
    def getBiGramProbability(self, w1, w2) -> float:
        if w1 in self.__children:
            child = self.__children[w1]
            return child.getUniGramProbability(w2)
        elif self.__unknown is not None:
            return self.__unknown.getUniGramProbability(w2)

    """
    Gets trigram probability of given symbols w1, w2 and w3.

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
        probability of given trigram.
    """
    def getTriGramProbability(self, w1, w2, w3) -> float:
        if w1 in self.__children:
            child = self.__children[w1]
            return child.getBiGramProbability(w2, w3)
        elif self.__unknown is not None:
            return self.__unknown.getBiGramProbability(w2, w3)

    """
    Counts words recursively given height and wordCounter.

    PARAMETERS
    ----------
    wordCounter : CounterHashMap
        word counter keeping symbols and their counts.
    height : int
        height for NGram. if height = 1, If height = 1, N-Gram is treated as UniGram, if height = 2, N-Gram is treated 
        as Bigram, etc.
    """
    def countWords(self, wordCounter: CounterHashMap, height: int):
        if height == 0:
            wordCounter.putNTimes(self.__symbol, self.__count)
        else:
            for child in self.__children.values():
                child.countWords(wordCounter, height - 1)

    """
    Replace words not in given dictionary.
    Deletes unknown words from children nodes and adds them to NGramNode#unknown unknown node as children recursively.

    PARAMETERS
    ----------
    dictionary : set
        dictionary of known words.
    """
    def replaceUnknownWords(self, dictionary: set):
        childList = []
        for symbol in self.__children.keys():
            if symbol not in dictionary:
                childList.append(self.__children[symbol])
        if len(childList) > 0:
            self.__unknown = NGramNode("")
            self.__unknown.__children = {}
            total = 0
            for child in childList:
                self.__unknown.__children.update(child.children)
                total += child.count
                del self.__children[child.symbol]
            self.__unknown.__count = total
            self.__unknown.replaceUnknownWords(dictionary)
        for child in self.__children.values():
            child.replaceUnknownWords(dictionary)

    """
    Gets count of symbol given array of symbols and index of symbol in this array.

    PARAMETERS
    ----------
    s : list
        array of symbols
    index : int
        index of symbol whose count is returned
        
    RETURNS
    -------
    int
        count of the symbol.
    """
    def getCountForListItem(self, s: list, index: int) -> int:
        if index < len(s):
            if s[index] in self.__children:
                return self.__children[s[index]].getCountForListItem(s, index + 1)
            else:
                return 0
        else:
            return self.getCount()

    """
    Generates next string for given list of symbol and index
    PARAMETERS
    ----------
    s : list
        array of symbols
    index : int
        index of generated string
        
    RETURNS
    -------
    object
        generated string.
    """
    def generateNextString(self, s: list, index: int) -> object:
        total = 0.0
        if index == len(s):
            prob = random.uniform(0, 1)
            for node in self.__children.values():
                if prob < node.probability + total:
                    return node.symbol
                else:
                    total += node.probability
        else:
            return self.__children[s[index]].generateNextString(s, index + 1)
        return None
