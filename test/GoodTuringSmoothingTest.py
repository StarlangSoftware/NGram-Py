import unittest

from NGram.GoodTuringSmoothing import GoodTuringSmoothing
from test.SimpleSmoothingTest import SimpleSmoothingTest


class GoodTuringSmoothingTest(SimpleSmoothingTest, unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        simpleSmoothing = GoodTuringSmoothing()
        self.simpleUniGram.calculateNGramProbabilitiesSimple(simpleSmoothing)
        self.simpleBiGram.calculateNGramProbabilitiesSimple(simpleSmoothing)
        self.simpleTriGram.calculateNGramProbabilitiesSimple(simpleSmoothing)
        self.complexUniGram.calculateNGramProbabilitiesSimple(simpleSmoothing)
        self.complexBiGram.calculateNGramProbabilitiesSimple(simpleSmoothing)
        self.complexTriGram.calculateNGramProbabilitiesSimple(simpleSmoothing)

    def test_PerplexitySimple(self):
        self.assertAlmostEqual(14.500734, self.simpleUniGram.getPerplexity(self.simpleCorpus), 4)
        self.assertAlmostEqual(2.762526, self.simpleBiGram.getPerplexity(self.simpleCorpus), 4)
        self.assertAlmostEqual(3.685001, self.simpleTriGram.getPerplexity(self.simpleCorpus), 4)

    def test_PerplexityComplex(self):
        self.assertAlmostEqual(1290.97916, self.complexUniGram.getPerplexity(self.testCorpus), 4)
        self.assertAlmostEqual(8331.518540, self.complexBiGram.getPerplexity(self.testCorpus), 4)
        self.assertAlmostEqual(39184.430078, self.complexTriGram.getPerplexity(self.testCorpus), 4)

    def test_CalculateNGramProbabilitiesSimple(self):
        self.assertAlmostEqual(0.116607, self.simpleUniGram.getProbability("<s>"), 4)
        self.assertAlmostEqual(0.149464, self.simpleUniGram.getProbability("mahmut"), 4)
        self.assertAlmostEqual(0.026599, self.simpleUniGram.getProbability("kitabı"), 4)
        self.assertAlmostEqual(0.492147, self.simpleBiGram.getProbability("<s>", "ali"), 4)
        self.assertAlmostEqual(0.030523, self.simpleBiGram.getProbability("ayşe", "ali"), 4)
        self.assertAlmostEqual(0.0625, self.simpleBiGram.getProbability("mahmut", "ali"), 4)
        self.assertAlmostEqual(0.323281, self.simpleBiGram.getProbability("at", "mehmet"), 4)
        self.assertAlmostEqual(0.049190, self.simpleTriGram.getProbability("<s>", "ali", "top"), 4)
        self.assertAlmostEqual(0.043874, self.simpleTriGram.getProbability("ayşe", "kitabı", "at"), 4)
        self.assertAlmostEqual(0.0625, self.simpleTriGram.getProbability("ayşe", "topu", "at"), 4)
        self.assertAlmostEqual(0.0625, self.simpleTriGram.getProbability("mahmut", "evde", "kal"), 4)
        self.assertAlmostEqual(0.261463, self.simpleTriGram.getProbability("ali", "topu", "at"), 4)

    def test_CalculateNGramProbabilitiesComplex(self):
        self.assertAlmostEqual(0.050745, self.complexUniGram.getProbability("<s>"), 4)
        self.assertAlmostEqual(0.000126, self.complexUniGram.getProbability("atatürk"), 4)
        self.assertAlmostEqual(0.000497, self.complexBiGram.getProbability("<s>", "mustafa"), 4)
        self.assertAlmostEqual(0.014000, self.complexBiGram.getProbability("mustafa", "kemal"), 4)
        self.assertAlmostEqual(0.061028, self.complexTriGram.getProbability("<s>", "mustafa", "kemal"), 4)
        self.assertAlmostEqual(0.283532, self.complexTriGram.getProbability("mustafa", "kemal", "atatürk"), 4)


if __name__ == '__main__':
    unittest.main()
