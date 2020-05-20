class CorpusTest:

    def readCorpus(self, fileName: str) -> list:
        corpus = []
        inputFile = open(fileName, "r")
        lines = inputFile.readlines()
        for line in lines:
            words = line.split(" ")
            corpus.append(words)
        inputFile.close()
        return corpus
