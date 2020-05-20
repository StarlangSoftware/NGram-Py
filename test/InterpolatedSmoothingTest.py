import unittest

from NGram.InterpolatedSmoothing import InterpolatedSmoothing
from test.SimpleSmoothingTest import SimpleSmoothingTest


class InterpolatedSmoothingTest(SimpleSmoothingTest, unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        interpolatedSmoothing = InterpolatedSmoothing()
        self.complexBiGram.calculateNGramProbabilitiesTrained(self.validationCorpus, interpolatedSmoothing)
        self.complexTriGram.calculateNGramProbabilitiesTrained(self.validationCorpus, interpolatedSmoothing)

    def test_PerplexityComplex(self):
        self.assertAlmostEqual(917.214864, self.complexBiGram.getPerplexity(self.testCorpus), 4)
        self.assertAlmostEqual(3000.451177, self.complexTriGram.getPerplexity(self.testCorpus), 4)

    def test_CalculateNGramProbabilitiesComplex(self):
        self.assertAlmostEqual(0.000418, self.complexBiGram.getProbability("<s>", "mustafa"), 4)
        self.assertAlmostEqual(0.005555, self.complexBiGram.getProbability("mustafa", "kemal"), 4)
        self.assertAlmostEqual(0.014406, self.complexTriGram.getProbability("<s>", "mustafa", "kemal"), 4)
        self.assertAlmostEqual(0.058765, self.complexTriGram.getProbability("mustafa", "kemal", "atatürk"), 4)


if __name__ == '__main__':
    unittest.main()
