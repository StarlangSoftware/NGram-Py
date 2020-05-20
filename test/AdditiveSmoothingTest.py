import unittest

from NGram.AdditiveSmoothing import AdditiveSmoothing
from test.SimpleSmoothingTest import SimpleSmoothingTest


class AdditiveSmoothingTest(SimpleSmoothingTest, unittest.TestCase):
    delta1: float
    delta2: float
    delta3: float

    def setUp(self) -> None:
        super().setUp()
        additiveSmoothing = AdditiveSmoothing()
        self.complexUniGram.calculateNGramProbabilitiesTrained(self.validationCorpus, additiveSmoothing)
        self.delta1 = additiveSmoothing.getDelta()
        self.complexBiGram.calculateNGramProbabilitiesTrained(self.validationCorpus, additiveSmoothing)
        self.delta2 = additiveSmoothing.getDelta()
        self.complexTriGram.calculateNGramProbabilitiesTrained(self.validationCorpus, additiveSmoothing)
        self.delta3 = additiveSmoothing.getDelta()

    def test_PerplexityComplex(self):
        self.assertAlmostEqual(4043.947022, self.complexUniGram.getPerplexity(self.testCorpus), 4)
        self.assertAlmostEqual(9220.218871, self.complexBiGram.getPerplexity(self.testCorpus), 4)
        self.assertAlmostEqual(30695.701941, self.complexTriGram.getPerplexity(self.testCorpus), 4)

    def test_CalculateNGramProbabilitiesComplex(self):
        self.assertEqual((20000 + self.delta1) / (376019.0 + self.delta1 * (self.complexUniGram.vocabularySize() + 1)), self.complexUniGram.getProbability("<s>"))
        self.assertEqual((50 + self.delta1) / (376019.0 + self.delta1 * (self.complexUniGram.vocabularySize() + 1)), self.complexUniGram.getProbability("atatürk"))
        self.assertEqual((11 + self.delta2) / (20000.0 + self.delta2 * (self.complexBiGram.vocabularySize() + 1)), self.complexBiGram.getProbability("<s>", "mustafa"))
        self.assertEqual((3 + self.delta2) / (138.0 + self.delta2 * (self.complexBiGram.vocabularySize() + 1)), self.complexBiGram.getProbability("mustafa", "kemal"))
        self.assertEqual((1 + self.delta3) / (11.0 + self.delta3 * (self.complexTriGram.vocabularySize() + 1)), self.complexTriGram.getProbability("<s>", "mustafa", "kemal"))
        self.assertEqual((1 + self.delta3) / (3.0 + self.delta3 * (self.complexTriGram.vocabularySize() + 1)), self.complexTriGram.getProbability("mustafa", "kemal", "atatürk"))


if __name__ == '__main__':
    unittest.main()
