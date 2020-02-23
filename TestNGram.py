from NGram.LaplaceSmoothing import LaplaceSmoothing
from NGram.NGram import NGram
from NGram.NoSmoothing import NoSmoothing

nGram = NGram(2)
nGram.addNGramSentence(["ali", "topu", "at", "mehmet", "ayşe", "gitti"])
nGram.addNGramSentence(["ali", "top", "at", "ayşe", "gitti"])
nGram.addNGramSentence(["ayşe", "kitabı", "ver"])
nGram.addNGramSentence(["ali", "topu", "mehmete", "at"])
nGram.addNGramSentence(["ali", "topu", "at", "mehmet", "ayşe", "gitti"])
nGram.calculateNGramProbabilitiesSimple(LaplaceSmoothing())
nGram.saveAsText("deneme.txt")
nGram2 = NGram("deneme.txt")
nGram2.saveAsText("deneme2.txt")