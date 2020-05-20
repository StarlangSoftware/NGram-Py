import unittest

from NGram.NGram import NGram
from test.CorpusTest import CorpusTest


class SimpleSmoothingTest(CorpusTest, unittest.TestCase):

    simpleUniGram: NGram
    simpleBiGram: NGram
    simpleTriGram: NGram
    complexUniGram: NGram
    complexBiGram: NGram
    complexTriGram: NGram
    simpleCorpus : list
    trainCorpus: list
    testCorpus: list
    validationCorpus: list

    def setUp(self) -> None:
        self.simpleCorpus = [["<s>", "ali", "topu", "at", "mehmet", "ayşeye", "gitti", "</s>"],
                             ["<s>", "ali", "top", "at", "ayşe", "eve", "gitti", "</s>"],
                             ["<s>", "ayşe", "kitabı", "ver", "</s>"],
                             ["<s>", "ali", "topu", "mehmete", "at", "</s>"],
                             ["<s>", "ali", "topu", "at", "mehmet", "ayşeyle", "gitti", "</s>"]]
        self.simpleUniGram = NGram(1, self.simpleCorpus)
        self.simpleBiGram = NGram(2, self.simpleCorpus)
        self.simpleTriGram = NGram(3, self.simpleCorpus)
        self.trainCorpus = self.readCorpus("../train.txt")
        self.complexUniGram = NGram(1, self.trainCorpus)
        self.complexBiGram = NGram(2, self.trainCorpus)
        self.complexTriGram = NGram(3, self.trainCorpus)
        self.testCorpus = self.readCorpus("../test.txt")
        self.validationCorpus = self.readCorpus("../validation.txt")
