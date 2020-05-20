import unittest

from NGram.NGram import NGram
from test.CorpusTest import CorpusTest


class MyTestCase(CorpusTest, unittest.TestCase):

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

    def test_GetCountSimple(self):
        self.assertEqual(5, self.simpleUniGram.getCount(["<s>"]))
        self.assertEqual(0, self.simpleUniGram.getCount(["mahmut"]), 0.0)
        self.assertEqual(1, self.simpleUniGram.getCount(["kitabı"]), 0.0)
        self.assertEqual(4, self.simpleBiGram.getCount(["<s>", "ali"]), 0.0)
        self.assertEqual(0, self.simpleBiGram.getCount(["ayşe", "ali"]), 0.0)
        self.assertEqual(0, self.simpleBiGram.getCount(["mahmut", "ali"]), 0.0)
        self.assertEqual(2, self.simpleBiGram.getCount(["at", "mehmet"]), 0.0)
        self.assertEqual(1, self.simpleTriGram.getCount(["<s>", "ali", "top"]), 0.0)
        self.assertEqual(0, self.simpleTriGram.getCount(["ayşe", "kitabı", "at"]), 0.0)
        self.assertEqual(0, self.simpleTriGram.getCount(["ayşe", "topu", "at"]), 0.0)
        self.assertEqual(0, self.simpleTriGram.getCount(["mahmut", "evde", "kal"]), 0.0)
        self.assertEqual(2, self.simpleTriGram.getCount(["ali", "topu", "at"]), 0.0)

    def test_GetCountComplex(self):
        self.assertEqual(20000, self.complexUniGram.getCount(["<s>"]), 0.0)
        self.assertEqual(50, self.complexUniGram.getCount(["atatürk"]), 0.0)
        self.assertEqual(11, self.complexBiGram.getCount(["<s>", "mustafa"]), 0.0)
        self.assertEqual(3, self.complexBiGram.getCount(["mustafa", "kemal"]), 0.0)
        self.assertEqual(1, self.complexTriGram.getCount(["<s>", "mustafa", "kemal"]), 0.0)
        self.assertEqual(1, self.complexTriGram.getCount(["mustafa", "kemal", "atatürk"]), 0.0)

    def test_VocabularySizeSimple(self):
        self.assertEqual(15, self.simpleUniGram.vocabularySize())

    def test_VocabularySizeComplex(self):
        self.assertEqual(57625, self.complexUniGram.vocabularySize(), 0.0)
        self.complexUniGram = NGram(1, self.testCorpus)
        self.assertEqual(55485, self.complexUniGram.vocabularySize(), 0.0)
        self.complexUniGram = NGram(1, self.validationCorpus)
        self.assertEqual(35663, self.complexUniGram.vocabularySize(), 0.0)

    def test_SaveAsText(self):
        self.simpleUniGram.saveAsText("simple1.txt")
        self.simpleBiGram.saveAsText("simple2.txt")
        self.simpleTriGram.saveAsText("simple3.txt")


if __name__ == '__main__':
    unittest.main()
