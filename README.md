For Developers
============

You can also see [Cython](https://github.com/starlangsoftware/NGram-Cy), [Java](https://github.com/starlangsoftware/NGram), [C++](https://github.com/starlangsoftware/NGram-CPP), [Swift](https://github.com/starlangsoftware/NGram-Swift), or [C#](https://github.com/starlangsoftware/NGram-CS) repository.

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
     
To create an empty NGram model:

	NGram(N: int)

For example,

	a = NGram(2)

this creates an empty NGram model.

To add an sentence to NGram

	addNGramSentence(self, symbols: list)

For example,

	nGram = NGram(2)
	nGram.addNGramSentence(["jack", "read", "books", "john", "mary", "went"])
	nGram.addNGramSentence(["jack", "read", "books", "mary", "went"])


with the lines above, an empty NGram model is created and two sentences are
added to the bigram model.

NoSmoothing class is the simplest technique for smoothing. It doesn't require training.
Only probabilities are calculated using counters. For example, to calculate the probabilities
of a given NGram model using NoSmoothing:

	a.calculateNGramProbabilities(NoSmoothing())

LaplaceSmoothing class is a simple smoothing technique for smoothing. It doesn't require
training. Probabilities are calculated adding 1 to each counter. For example, to calculate
the probabilities of a given NGram model using LaplaceSmoothing:

	a.calculateNGramProbabilities(LaplaceSmoothing())

GoodTuringSmoothing class is a complex smoothing technique that doesn't require training.
To calculate the probabilities of a given NGram model using GoodTuringSmoothing:

	a.calculateNGramProbabilities(GoodTuringSmoothing())

AdditiveSmoothing class is a smoothing technique that requires training.

	a.calculateNGramProbabilities(AdditiveSmoothing())

## Using NGram

To find the probability of an NGram:

	getProbability(self, *args) -> float

For example, to find the bigram probability:

	a.getProbability("jack", "reads")

To find the trigram probability:

	a.getProbability("jack", "reads", "books")

## Saving NGram
    
To save the NGram model:

	saveAsText(self, fileName: str)

For example, to save model "a" to the file "model.txt":

	a.saveAsText("model.txt");              

## Loading NGram            

To load an existing NGram model:

	NGram(fileName: str)

For example,

	a = NGram("model.txt")

this loads an NGram model in the file "model.txt".
