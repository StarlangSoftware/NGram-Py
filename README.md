# NGram

For Developers
============
You can also see [Java](https://github.com/starlangsoftware/NGram), [C++](https://github.com/starlangsoftware/NGram-CPP), or [C#](https://github.com/starlangsoftware/NGram-CS) repository.

## Requirements

* [Python 3.7 or higher](#python)
* [Git](#git)

### Python 

To check if you have a compatible version of Python installed, use the following command:

    python -V
    
You can find the latest version of Python [here](https://www.python.org/downloads/).

### Git

Install the [latest version of Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

## Download Code

In order to work on code, create a fork from GitHub page. 
Use Git for cloning the code to your local or below line for Ubuntu:

	git clone <your-fork-git-link>

A directory called NGram will be created. Or you can use below link for exploring the code:

	git clone https://github.com/starlangsoftware/NGram-Py.git

## Open project with Pycharm IDE

Steps for opening the cloned project:

* Start IDE
* Select **File | Open** from main menu
* Choose `NGram-PY` file
* Select open as project option
* Couple of seconds, dependencies will be downloaded. 

Detailed Description
============
+ [Training NGram](#training-ngram)
+ [Using NGram](#using-ngram)
+ [Saving NGram](#saving-ngram)
+ [Loading NGram](#loading-ngram)

## Training NGram
     
Boş bir NGram modeli oluşturmak için

	NGram(N: int)

Örneğin,

	a = NGram(2)

boş bir bigram modeli oluşturulmaktadır.

NGram'a bir cümle eklemek için

	addNGramSentence(self, symbols: list)

Örneğin,

	nGram = NGram(2)
	nGram.addNGramSentence(["ali", "topu", "at", "mehmet", "ayşe", "gitti"])
	nGram.addNGramSentence(["ali", "top", "at", "ayşe", "gitti"])


satırları ile boş bir bigram oluşturulup, iki cümle bigram modeline 
eklenir.

NoSmoothing sınıfı smoothing için kullanılan en basit tekniktir. Eğitim gerektirmez, sadece
sayaçlar kullanılarak olasılıklar hesaplanır. Örneğin verilen bir NGram'ın NoSmoothing ile 
olasılıklarının hesaplanması için

	a.calculateNGramProbabilities(NoSmoothing())

LaplaceSmoothing sınıfı smoothing için kullanılan basit bir yumuşatma tekniğidir. Eğitim 
gerektirmez, her sayaca 1 eklenerek olasılıklar hesaplanır. Örneğin verilen bir NGram'ın 
LaplaceSmoothing ile olasılıklarının hesaplanması için

	a.calculateNGramProbabilities(LaplaceSmoothing())

GoodTuringSmoothing sınıfı smoothing için kullanılan eğitim gerektirmeyen karmaşık bir 
yumuşatma tekniğidir. Verilen bir NGram'ın GoodTuringSmoothing ile olasılıklarının 
hesaplanması için

	a.calculateNGramProbabilities(GoodTuringSmoothing())

AdditiveSmoothing sınıfı smoothing için kullanılan eğitim gerektiren bir yumuşatma 
tekniğidir.

	a.calculateNGramProbabilities(AdditiveSmoothing())

## Using NGram

Bir NGram'ın olasılığını bulmak için

	getProbability(self, *args) -> float

Örneğin, bigram olasılığını bulmak için

	a.getProbability("ali", "topu")

trigram olasılığını bulmak için

	a.getProbability("ali", "topu", "at")

## Saving NGram
    
NGram modelini kaydetmek için

	saveAsText(self, fileName: str)

Örneğin, a modelini "model.txt" dosyasına kaydetmek için

	a.saveAsText("model.txt");              

## Loading NGram            

Var olan bir NGram modelini yüklemek için

	NGram(fileName: str)

Örneğin,

	a = NGram("model.txt")

model.txt dosyasında bulunan bir NGram modelini yükler.

