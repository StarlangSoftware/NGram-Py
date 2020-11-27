from __future__ import annotations

from io import TextIOWrapper
from DataStructure.CounterHashMap import CounterHashMap
from NGram.MultipleFile import MultipleFile
import random


class NGramNode(object):

    __children: dict
    __symbol: object
    __count: int
    __probability: float
    __probabilityOfUnseen: float
    __unknown: NGramNode

    def __init__(self, symbolOrIsRootNode, inputFile=None):
        """
        Constructor of NGramNode

        PARAMETERS
        ----------
        symbolOrIsRootNode
            symbol to be kept in this node.
        """
        self.__unknown = None
        if not isinstance(symbolOrIsRootNode, bool):
            self.__symbol = symbolOrIsRootNode
            self.__count = 0
            self.__probability = 0.0
            self.__probabilityOfUnseen = 0.0
            self.__children = {}
        else:
            if isinstance(symbolOrIsRootNode, bool) and inputFile is not None:
                if isinstance(inputFile, TextIOWrapper):
                    if not symbolOrIsRootNode:
                        self.__symbol = inputFile.readline().strip()
                    line = inputFile.readline().strip()
                    items = line.split()
                    self.__count = int(items[0])
                    self.__probability = float(items[1])
                    self.__probabilityOfUnseen = float(items[2])
                    numberOfChildren = int(items[3])
                    if numberOfChildren > 0:
                        self.__children = {}
                        for i in range(numberOfChildren):
                            childNode = NGramNode(False, inputFile)
                            self.__children[childNode.__symbol] = childNode
                    else:
                        self.__children = {}
                elif isinstance(inputFile, MultipleFile):
                    if not symbolOrIsRootNode:
                        self.__symbol = inputFile.readLine().strip()
                    line = inputFile.readLine().strip()
                    items = line.split()
                    self.__count = int(items[0])
                    self.__probability = float(items[1])
                    self.__probabilityOfUnseen = float(items[2])
                    numberOfChildren = int(items[3])
                    if numberOfChildren > 0:
                        self.__children = {}
                        for i in range(numberOfChildren):
                            childNode = NGramNode(False, inputFile)
                            self.__children[childNode.__symbol] = childNode
                    else:
                        self.__children = {}

    def getCount(self) -> int:
        """
        Gets count of this node.

        RETURNS
        -------
        int
            count of this node.
        """
        return self.__count

    def size(self) -> int:
        """
        Gets the size of children of this node.

        RETURNS
        -------
        int
            size of children of NGramNode this node.
        """
        return len(self.__children)

    def maximumOccurence(self, height: int) -> int:
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
        maxValue = 0
        if height == 0:
            return self.__count
        else:
            for child in self.__children.values():
                current = child.maximumOccurence(height - 1)
                if current > maxValue:
                    maxValue = current
            return maxValue

    def childSum(self) -> float:
        """
        RETURNS
        -------
        float
            sum of counts of children nodes.
        """
        total = 0
        for child in self.__children.values():
            total += child.__count
        if self.__unknown is not None:
            total += self.__unknown.__count
        return total

    def updateCountsOfCounts(self, countsOfCounts: list, height: int):
        """
        Traverses nodes and updates counts of counts for each node.

        PARAMETERS
        ----------
        countsOfCounts : list
            counts of counts of NGrams.
        height : int
            height for NGram. if height = 1, If level = 1, N-Gram is treated as UniGram, if level = 2, N-Gram is treated
            as Bigram, etc.
        """
        if height == 0:
            countsOfCounts[self.__count] = countsOfCounts[self.__count] + 1
        else:
            for child in self.__children.values():
                child.updateCountsOfCounts(countsOfCounts, height - 1)

    def setProbabilityWithPseudoCount(self, pseudoCount: float, height: int, vocabularySize: float):
        """
        Sets probabilities by traversing nodes and adding pseudocount for each NGram.

        PARAMETERS
        ----------
        pseudoCount : int
            pseudocount added to each NGram.
        height : int
            height for NGram. if height = 1, If level = 1, N-Gram is treated as UniGram, if level = 2, N-Gram is treated
            as Bigram, etc.
        vocabularySize : float
            size of vocabulary
        """
        if height == 1:
            total = self.childSum() + pseudoCount * vocabularySize
            for child in self.__children.values():
                child.__probability = (child.__count + pseudoCount) / total
            if self.__unknown is not None:
                self.__unknown.__probability = (self.__unknown.__count + pseudoCount) / total
            self.__probabilityOfUnseen = pseudoCount / total
        else:
            for child in self.__children.values():
                child.setProbabilityWithPseudoCount(pseudoCount, height - 1, vocabularySize)

    def setAdjustedProbability(self, N: list, height: int, vocabularySize: float, pZero: float):
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
            height for NGram. if height = 1, If level = 1, N-Gram is treated as UniGram, if level = 2, N-Gram is treated
            as Bigram, etc.
        vocabularySize : float
            size of vocabulary.
        pZero : float
            probability of zero.
        """
        if height == 1:
            total = 0
            for child in self.__children.values():
                r = child.__count
                if r <= 5:
                    newR = ((r + 1) * N[r + 1]) / N[r]
                    total += newR
                else:
                    total += r
            for child in self.__children.values():
                r = child.__count
                if r <= 5:
                    newR = ((r + 1) * N[r + 1]) / N[r]
                    child.__probability = (1 - pZero) * (newR / total)
                else:
                    child.__probability = (1 - pZero) * (r / total)
            self.__probabilityOfUnseen = pZero / (vocabularySize - len(self.__children))
        else:
            for child in self.__children.values():
                child.setAdjustedProbability(N, height - 1, vocabularySize, pZero)

    def addNGram(self, s: list, index: int, height: int, sentenceCount: int = 1):
        """
        Adds NGram given as array of symbols to the node as a child.

        PARAMETERS
        ----------
        s : list
            array of symbols
        index : int
            start index of NGram
        height : int
            height for NGram. if height = 1, If level = 1, N-Gram is treated as UniGram, if level = 2, N-Gram is treated
            as Bigram, etc.
        sentenceCount : int
            Number of times this sentence is added.
        """
        if height == 0:
            return
        symbol = s[index]
        if symbol in self.__children:
            child = self.__children[symbol]
        else:
            child = NGramNode(symbol)
            self.__children[symbol] = child
        child.__count += sentenceCount
        child.addNGram(s, index + 1, height - 1, sentenceCount)

    def getUniGramProbability(self, w1) -> float:
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
        if w1 in self.__children:
            return self.__children[w1].__probability
        elif self.__unknown is not None:
            return self.__unknown.__probability
        else:
            return self.__probabilityOfUnseen

    def getBiGramProbability(self, w1, w2) -> float:
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
        if w1 in self.__children:
            child = self.__children[w1]
            return child.getUniGramProbability(w2)
        elif self.__unknown is not None:
            return self.__unknown.getUniGramProbability(w2)
        else:
            return None

    def getTriGramProbability(self, w1, w2, w3) -> float:
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
        if w1 in self.__children:
            child = self.__children[w1]
            return child.getBiGramProbability(w2, w3)
        elif self.__unknown is not None:
            return self.__unknown.getBiGramProbability(w2, w3)
        else:
            return None

    def countWords(self, wordCounter: CounterHashMap, height: int):
        """
        Counts words recursively given height and wordCounter.

        PARAMETERS
        ----------
        wordCounter : CounterHashMap
            word counter keeping symbols and their counts.
        height : int
            height for NGram. if height = 1, If height = 1, N-Gram is treated as UniGram, if height = 2, N-Gram is
            treated as Bigram, etc.
        """
        if height == 0:
            wordCounter.putNTimes(self.__symbol, self.__count)
        else:
            for child in self.__children.values():
                child.countWords(wordCounter, height - 1)

    def replaceUnknownWords(self, dictionary: set):
        """
        Replace words not in given dictionary.
        Deletes unknown words from children nodes and adds them to NGramNode#unknown unknown node as children
        recursively.

        PARAMETERS
        ----------
        dictionary : set
            dictionary of known words.
        """
        childList = []
        for symbol in self.__children.keys():
            if symbol not in dictionary:
                childList.append(self.__children[symbol])
        if len(childList) > 0:
            self.__unknown = NGramNode("")
            self.__unknown.__children = {}
            total = 0
            for child in childList:
                self.__unknown.__children.update(child.__children)
                total += child.__count
                del self.__children[child.symbol]
            self.__unknown.__count = total
            self.__unknown.replaceUnknownWords(dictionary)
        for child in self.__children.values():
            child.replaceUnknownWords(dictionary)

    def getCountForListItem(self, s: list, index: int) -> int:
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
        if index < len(s):
            if s[index] in self.__children:
                return self.__children[s[index]].getCountForListItem(s, index + 1)
            else:
                return 0
        else:
            return self.getCount()

    def generateNextString(self, s: list, index: int) -> object:
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
        total = 0.0
        if index == len(s):
            prob = random.uniform(0, 1)
            for node in self.__children.values():
                if prob < node.__probability + total:
                    return node.symbol
                else:
                    total += node.__probability
        else:
            return self.__children[s[index]].generateNextString(s, index + 1)
        return None

    def prune(self, threshold: float, N: int):
        if N == 0:
            toBeDeleted = []
            for symbol in self.__children.keys():
                if self.__children[symbol].__count / self.__count < threshold:
                    toBeDeleted.append(symbol)
            for symbol in toBeDeleted:
                self.__children.pop(symbol)
        else:
            for node in self.__children.values():
                node.prune(threshold, N - 1)

    def saveAsText(self, isRootNode: bool, outputFile, level: int):
        """
        Save this NGramNode to a text file.

        PARAMETERS
        ----------
        isRootNode: bool
            True if this not is a root node, false otherwise
        outputFile
            file where NGram is saved.
        level: int
            Level of this node
        """
        if not isRootNode:
            for i in range(level):
                outputFile.write("\t")
            outputFile.write(self.__symbol.__str__() + "\n")
        for i in range(level):
            outputFile.write("\t")
        if len(self.__children) > 0:
            outputFile.write(self.__count.__str__() + " " + self.__probability.__str__() + " " +
                             self.__probabilityOfUnseen.__str__() + " " + self.size().__str__() + "\n")
            for child in self.__children.values():
                child.saveAsText(False, outputFile, level + 1)
        else:
            outputFile.write(self.__count.__str__() + " " + self.__probability.__str__() + " " +
                             self.__probabilityOfUnseen.__str__() + " 0\n")
