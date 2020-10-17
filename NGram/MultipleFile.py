class MultipleFile:

    index: int
    fileNameList: list
    lines: list
    lineIndex: int

    def __init__(self, fileList: list):
        self.fileNameList = fileList
        file = open(self.fileNameList[0])
        self.lines = file.readlines()
        self.lineIndex = 0
        self.index = 0

    def readLine(self) -> str:
        if self.lineIndex == len(self.lines):
            self.index = self.index + 1
            file = open(self.fileNameList[self.index])
            self.lines = file.readlines()
            self.lineIndex = 0
        line = self.lines[self.lineIndex]
        self.lineIndex = self.lineIndex + 1
        return line
