<h1 align="center">Personalized Medicine</h1>

## Project Overview

[Personalized Medicine](https://en.wikipedia.org/wiki/Personalized_medicine#:~:text=Personalized%20medicine%2C%20also%20referred%20to,response%20or%20risk%20of%20disease.) (PM) [1] is promising in improving health care efficiently and safely because it makes it possible for patients to receive earlier diagnoses, risk assessments, and optimal treatments. However, PM in cancer treatment is still slowly developing due to a large amount of textual content based-medical literature and clinical observations needed to be manually analyzed. To address this issue and speed up the progress of PM, we propose an efficient text classifier to automatically classify the effect of genetic variants. We show that contributions from NLP community and specialists can definitely bring PM into a bright future.

## Methods

### Data

The data comes from [Kaggle](https://www.kaggle.com/c/msk-redefining-cancer-treatment/overview) competition and Memorial Sloan Kettering Cancer Center.

Two data files were used in this project. There are 3321 data points in total. `Gene` and `Variation` are categorical.  `Class` contains 9 unique values.`TEXT` contains biomedical literature related to each class/ effect of genetic variants.

* training_text.zip: `ID` `TEXT`
* training_variants.zip: `ID` `Gene` `Variation` `Class` 

### Data Preprocessing

* Tools: [NLTK](https://www.nltk.org/), [SpaCy](https://spacy.io/) 

* Steps: 

  * Tokenization
  * Removal of punctuations
  * Lemmatization
  * Removal of stop words
  * Removal of integers, and single integer-single character combinations (Special consideration for biomedical literature)
  * Lower casting 

  (Stop words were removed for building machine learning classifiers, but not for building neural networks, as neural networks are trying to learn the semantic meaning and the meaning of a word depends on the context.)

* Special consideration for biomedical literature: 
  * In addition to removing suggested stop words by tools, also remove those common and meaningless in biomedical literature, such as 'figure', 'fig', 'table','tab', 'supplement', 'supplementary', 'download','author', 'et', 'al', etc.
  * Remove integers that are not aside any character (intext citation numbers).
  * Remove single integer - single character combinations (figure index in literature), such as '1a', '7c', etc.
  
* Any null content of the text was replaced by the merged string of gene mutation type and variation type of the same row. Any space that appeared in the Gene and Variation columns was replaced by an underscore. 

### Exploratory Data Analysis

* Univariant and bivariant analyses to show the distribution of categorical variables and the interaction between them. 
* The distribution of words and characters in all text data. The distribution of unigram, bigram, and trigram for each class. And word cloud plots for each class.

[Here](https://github.com/JoKerDii/Personalized-Medicine/tree/master/exploratory-data-analysis) is the EDA demo and some interesting findings. 

### Feature Extraction 

Methods:

* One-hot encoding gene and variation features

* Bag-of-Words (Tools: Count vectorizer and TFIDF vectorizer) and Word Embeddings (Source:  [stanford GloVe](https://nlp.stanford.edu/projects/glove/), [BioWordVec](https://github.com/ncbi-nlp/BioWordVec), [BioSentVec](https://github.com/ncbi-nlp/BioSentVec), [BioConceptVec](https://github.com/ncbi/BioConceptVec), [pubmed2018_w2v_200D](https://ia803102.us.archive.org/13/items/pubmed2018_w2v_200D.tar/README.txt), [pubmed2018_w2v_400D](https://github.com/RaRe-Technologies/gensim-data/issues/28)) for text feature.
* A pre-trained model trained on PubMed biomedicine literature by [Microsoft](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext#) for BERT model.

Features for training machine learning models: one-hot gene and variation + SVD truncated count vector and TFIDF vector of text. 

Features for training neural networks: feature vectors transformed by `pubmed2018_w2v_400D` pre-trained model (Except for BERT).

[Here](https://github.com/JoKerDii/Personalized-Medicine/tree/master/word-embedding-and-bow) is an evaluation of some text feature extraction methods.

### Machine Learning Methods（Baseline）

Eight supervised machine learning methods were applied: 

* Support Vector Machine (SVM)
* Logistic Regression (LR)
* k-Nearest Neighbors (kNN)
* Random Forest (RF)
* Adaptive Boosting (AdaBoost)
* eXtreme Gradient Boosting (XGBoost)
* Multi-Laryer Perceptron (MLP)
* An ensemble (voting) model of LR, RF, KNN, and SVM

[Here](https://github.com/JoKerDii/Personalized-Medicine/tree/master/eight-ml-classifiers) is an evaluation of model performance.

### Neural Networks

**CNN** [2] and **BiLSTM** [3].  RCNN, RNN+Attention, and BERT are worth trying.

Pre-trained word-embedding is chosen from: [pubmed_w2v_400D](https://github.com/RaRe-Technologies/gensim-data/issues/28)

### Evaluation Metrics

* Log loss / Cross entropy loss
* Balanced accuracy
* F1-score (micro-average)

## Results and Discussions

* [An EDA demo](https://github.com/JoKerDii/Personalized-Medicine/blob/master/exploratory-data-analysis/eda-demo.ipynb)
* [Comparison of over sampling techniques](https://github.com/JoKerDii/Personalized-Medicine/blob/master/exploratory-data-analysis/resampling.ipynb)
* It seems traditional over sampling cannot solve the imbalanced data problem. Even worse, over sampling could introduce serious overfitting.
* [Comparison of pre-trained word embedding models](https://github.com/JoKerDii/Personalized-Medicine/tree/master/word-embedding-and-bow)
  * The representative power of pre-trained word embedding model highly depends on the dataset itself.
* [Evaluation of eight machine learning methods](https://github.com/JoKerDii/Personalized-Medicine/blob/master/eight-ml-classifiers/performance-of-ml-classifiers.ipynb)
* [Evaluation of several neural network -based models](https://github.com/JoKerDii/Personalized-Medicine/tree/master/neural-nets)

## Future Work

1. Include different types of dataset from other sources, e.g, personal information (family disease history, age, race, etc). 
2. Upcome other three or more NN based models.
3. Combine / stack NN models.
4. Build Non-static NN models. It is reported that non-static NN models are always better than static NN models.
5. Concatenate multiple word vector representations (e.g. pubmed_w2c_400D and BioConceptVec). In addition to word vectors trained from PubMed, biological concepts are important features.
6. Deal with imbalanced text data by sentence / word augmentation using [nlpaug](https://github.com/makcedward/nlpaug).
7. Build a hybrid model / multi-model: one part trained on text data, the other trained on sequence data to capture genetic variants ( like what [DeepSEA](https://github.com/SUSE/DeepSea) does)

## Environment

python 3.8

pytorch 1.7.0

## Directory

````
```
Personalized-Medicine
├── eight-ml-classifiers
│   ├── images
│   │   ├── confusion-matrix
│   │   ├── learning-curve
│   │   ├── Accuracy_allmodel.png
│   │   ├── F1score_allmodels.png
│	│   └── logloss_allmodels.png
│   ├── README.md
│	├── data-preprocessing_v1.py
│	├── data-preprocessing_v2.py
│	├── feature-extraction.py
│	├── model-evaluation.py
│	├── performance-of-ml-classifiers.ipynb
│	├── train-models.py
│	├── workflow-part1.ipynb
│	└── workflow-part2.ipynb
├── exploratory-data analysis
│   ├── images
│   │   ├── other_distribution
│   │   │   ├── dist_char.png
│   │   │   ├── dist_class.png
│   │   │   ├── dist_gene.png
│   │   │   ├── dist_variation.png
│   │   │   ├── dist_word.png
│   │   │   ├── gene_class.png
│   │   │   └── word_class.png
│   │   ├── uni_bi_trigram_distribution
│   │   │   ├── bi_c1.png
│   │   │   ├── ...
│   │   │   ├── bi_c9.png
│   │   │   ├── tri_c1.png
│   │   │   ├── ...
│   │   │   ├── tri_c9.png
│   │   │   ├── uni_c1.png
│   │   │   ├── ...
│   │   │   └── uni_c9.png
│   │   └── wordcloud_image
│   │   │   ├── wordCloud_class_1.png
│   │   │   ├── ...
│   │   │   ├── wordCloud_class_9.png
│   │   │   ├── wordCloud_not_strict.png
│   │   │   └── wordCloud_strict.png
│   ├── eda-demo.ipynb
│   ├── eda-gene-variation.py
│   ├── eda-text.py
│   └── resampling.ipynb
├── neural-nets
│   ├── image
│   │   ├── CE
│   │   ├── acc
│   │   ├── cm
│   │   ├── f1score
│   │   └── logloss
│   ├── models
│   │   ├── __pycache__
│   │   ├── CNN.py
│   │   └── BiLSTM.py
│   ├── LICENSE
│   ├── run.py
│   ├── train_eval.py
│   ├── utils.py
│   └── visualize.py
├── word-embedding-and-bow
│   ├── README.md
│   ├── bioconceptvec-rf.py
│   ├── biosentvec-rf.py
│   ├── biowordvec-rf.py
│   ├── glove-rf.py
│   ├── tfidf-count-rf.py
│   └── word2vec-rf.py
├── LICENSE.txt
└── README.md

​```
````



## References

[1] [Personalized Medicine: Part 1: Evolution and Development into Theranostics](https://pubmed.ncbi.nlm.nih.gov/21037908/) 
[2] [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)  
[3] [Recurrent Neural Network for Text Classification with Multi-Task Learning](https://arxiv.org/abs/1605.05101)  

