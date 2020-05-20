import unittest

from NGram.NoSmoothing import NoSmoothing
from test.SimpleSmoothingTest import SimpleSmoothingTest


class NoSmoothingTest(SimpleSmoothingTest, unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        simpleSmoothing = NoSmoothing()
        self.simpleUniGram.calculateNGramProbabilitiesSimple(simpleSmoothing)
        self.simpleBiGram.calculateNGramProbabilitiesSimple(simpleSmoothing)
        self.simpleTriGram.calculateNGramProbabilitiesSimple(simpleSmoothing)
        self.complexUniGram.calculateNGramProbabilitiesSimple(simpleSmoothing)
        self.complexBiGram.calculateNGramProbabilitiesSimple(simpleSmoothing)
        self.complexTriGram.calculateNGramProbabilitiesSimple(simpleSmoothing)

    def test_PerplexitySimple(self):
        self.assertAlmostEqual(12.318362, self.simpleUniGram.getPerplexity(self.simpleCorpus), 4)
        self.assertAlmostEqual(1.573148, self.simpleBiGram.getPerplexity(self.simpleCorpus), 4)
        self.assertAlmostEqual(1.248330, self.simpleTriGram.getPerplexity(self.simpleCorpus), 4)

    def test_PerplexityComplex(self):
        self.assertAlmostEqual(3220.299369, self.complexUniGram.getPerplexity(self.trainCorpus), 4)
        self.assertAlmostEqual(32.362912, self.complexBiGram.getPerplexity(self.trainCorpus), 4)
        self.assertAlmostEqual(2.025259, self.complexTriGram.getPerplexity(self.trainCorpus), 4)

    def test_CalculateNGramProbabilitiesSimple(self):
        self.assertEqual(5 / 35.0, self.simpleUniGram.getProbability("<s>"))
        self.assertEqual(0.0, self.simpleUniGram.getProbability("mahmut"))
        self.assertEqual(1.0 / 35.0, self.simpleUniGram.getProbability("kitabı"))
        self.assertEqual(4 / 5.0, self.simpleBiGram.getProbability("<s>", "ali"))
        self.assertEqual(0 / 2.0, self.simpleBiGram.getProbability("ayşe", "ali"))
        self.assertEqual(0.0, self.simpleBiGram.getProbability("mahmut", "ali"))
        self.assertEqual(2 / 4.0, self.simpleBiGram.getProbability("at", "mehmet"))
        self.assertEqual(1 / 4.0, self.simpleTriGram.getProbability("<s>", "ali", "top"))
        self.assertEqual(0 / 1.0, self.simpleTriGram.getProbability("ayşe", "kitabı", "at"))
        self.assertEqual(0.0, self.simpleTriGram.getProbability("ayşe", "topu", "at"))
        self.assertEqual(0.0, self.simpleTriGram.getProbability("mahmut", "evde", "kal"))
        self.assertEqual(2 / 3.0, self.simpleTriGram.getProbability("ali", "topu", "at"))

    def test_CalculateNGramProbabilitiesComplex(self):
        self.assertEqual(20000 / 376019.0, self.complexUniGram.getProbability("<s>"))
        self.assertEqual(50 / 376019.0, self.complexUniGram.getProbability("atatürk"))
        self.assertEqual(11 / 20000.0, self.complexBiGram.getProbability("<s>", "mustafa"))
        self.assertEqual(3 / 138.0, self.complexBiGram.getProbability("mustafa", "kemal"))
        self.assertEqual(1 / 11.0, self.complexTriGram.getProbability("<s>", "mustafa", "kemal"))
        self.assertEqual(1 / 3.0, self.complexTriGram.getProbability("mustafa", "kemal", "atatürk"))


if __name__ == '__main__':
    unittest.main()
