# Sentiment analysis
Machine learning models for sentiment analysis on tweets. 

## To run the code:
``` bash
$ python main.py <name_experiment>
```
## To add an experiment:
Just add a classifier and a feature extractor classes in the directory `experiments`.

## To get the data:
Refer to <a href="https://www.cs.york.ac.uk/semeval-2013/task2/index.php%3Fid=data.html">this webpage</a>. 

## Dependencies:
Depending on the experiments, one may need: pandas, sklearn, nltk, keras, gensim, numpy, re, pytorch.

We use stopwords.words('english') from nltk.corpus. To get this stopwords corpus, use the NLTK Downloader.
Open a Python console and do the following:
``` bash
>>> import nltk
>>> nltk.download()
```

## Pretrained embeddings
For the RecursiveNN and CNN parts, we use pretrained embeddings.
One can get these files (.zip) at https://nlp.stanford.edu/projects/glove/, unzip them and put them in the experiments folder. 
