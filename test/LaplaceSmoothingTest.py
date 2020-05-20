import unittest

from NGram.LaplaceSmoothing import LaplaceSmoothing
from test.SimpleSmoothingTest import SimpleSmoothingTest


class LaplaceSmoothingTest(SimpleSmoothingTest, unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        simpleSmoothing = LaplaceSmoothing()
        self.simpleUniGram.calculateNGramProbabilitiesSimple(simpleSmoothing)
        self.simpleBiGram.calculateNGramProbabilitiesSimple(simpleSmoothing)
        self.simpleTriGram.calculateNGramProbabilitiesSimple(simpleSmoothing)
        self.complexUniGram.calculateNGramProbabilitiesSimple(simpleSmoothing)
        self.complexBiGram.calculateNGramProbabilitiesSimple(simpleSmoothing)
        self.complexTriGram.calculateNGramProbabilitiesSimple(simpleSmoothing)

    def test_PerplexitySimple(self):
        self.assertAlmostEqual(12.809502, self.simpleUniGram.getPerplexity(self.simpleCorpus), 4)
        self.assertAlmostEqual(6.914532, self.simpleBiGram.getPerplexity(self.simpleCorpus), 4)
        self.assertAlmostEqual(7.694528, self.simpleTriGram.getPerplexity(self.simpleCorpus), 4)

    def test_PerplexityComplex(self):
        self.assertAlmostEqual(4085.763010, self.complexUniGram.getPerplexity(self.testCorpus), 4)
        self.assertAlmostEqual(24763.660225, self.complexBiGram.getPerplexity(self.testCorpus), 4)
        self.assertAlmostEqual(49579.187475, self.complexTriGram.getPerplexity(self.testCorpus), 4)

    def test_CalculateNGramProbabilitiesSimple(self):
        self.assertEqual((5 + 1) / (35 + self.simpleUniGram.vocabularySize() + 1), self.simpleUniGram.getProbability("<s>"))
        self.assertEqual((0 + 1) / (35 + self.simpleUniGram.vocabularySize() + 1), self.simpleUniGram.getProbability("mahmut"))
        self.assertEqual((1 + 1) / (35 + self.simpleUniGram.vocabularySize() + 1), self.simpleUniGram.getProbability("kitabı"))
        self.assertEqual((4 + 1) / (5 + self.simpleBiGram.vocabularySize() + 1), self.simpleBiGram.getProbability("<s>", "ali"))
        self.assertEqual((0 + 1) / (2 + self.simpleBiGram.vocabularySize() + 1), self.simpleBiGram.getProbability("ayşe", "ali"))
        self.assertEqual(1 / (self.simpleBiGram.vocabularySize() + 1), self.simpleBiGram.getProbability("mahmut", "ali"))
        self.assertEqual((2 + 1) / (4 + self.simpleBiGram.vocabularySize() + 1), self.simpleBiGram.getProbability("at", "mehmet"))
        self.assertEqual((1 + 1) / (4.0 + self.simpleTriGram.vocabularySize() + 1), self.simpleTriGram.getProbability("<s>", "ali", "top"))
        self.assertEqual((0 + 1) / (1.0 + self.simpleTriGram.vocabularySize() + 1), self.simpleTriGram.getProbability("ayşe", "kitabı", "at"))
        self.assertEqual(1 / (self.simpleTriGram.vocabularySize() + 1), self.simpleTriGram.getProbability("ayşe", "topu", "at"))
        self.assertEqual(1 / (self.simpleTriGram.vocabularySize() + 1), self.simpleTriGram.getProbability("mahmut", "evde", "kal"))
        self.assertEqual((2 + 1) / (3.0 + self.simpleTriGram.vocabularySize() + 1), self.simpleTriGram.getProbability("ali", "topu", "at"))

    def test_CalculateNGramProbabilitiesComplex(self):
        self.assertEqual((20000 + 1) / (376019.0 + self.complexUniGram.vocabularySize() + 1), self.complexUniGram.getProbability("<s>"))
        self.assertEqual((50 + 1) / (376019.0 + self.complexUniGram.vocabularySize() + 1), self.complexUniGram.getProbability("atatürk"))
        self.assertEqual((11 + 1) / (20000.0 + self.complexBiGram.vocabularySize() + 1), self.complexBiGram.getProbability("<s>", "mustafa"))
        self.assertEqual((3 + 1) / (138.0 + self.complexBiGram.vocabularySize() + 1), self.complexBiGram.getProbability("mustafa", "kemal"))
        self.assertEqual((1 + 1) / (11.0 + self.complexTriGram.vocabularySize() + 1), self.complexTriGram.getProbability("<s>", "mustafa", "kemal"))
        self.assertEqual((1 + 1) / (3.0 + self.complexTriGram.vocabularySize() + 1), self.complexTriGram.getProbability("mustafa", "kemal", "atatürk"))


if __name__ == '__main__':
    unittest.main()
