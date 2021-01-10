<h1 align="center">Personalized Medicine</h1>

This is basically a multi-class text classification task. My first journey of Natural Language Processing (NLP)  :P

## Project

[Personalized Medicine](https://en.wikipedia.org/wiki/Personalized_medicine#:~:text=Personalized%20medicine%2C%20also%20referred%20to,response%20or%20risk%20of%20disease.) (PM) is promising to improve health care in both efficiency and safety. It enables each patient to receive earlier diagnoses, risk assessments, and optimal treatments, by using individual genetic profiles. However, PM in cancer treatment is still slowly developing due to a large amount of textual content based-medical literature and clinical observations to be manually analyzed. To address this issue and speed up the progress of PM, we need an efficient text classifier to automatically classify the effect of genetic variants. 

The idea and data are from Kaggle competition on topic of '[Personalized Medicine: Redefining Cancer Treatment](https://www.kaggle.com/c/msk-redefining-cancer-treatment/overview)'. In my view, although the competition was flawed due to data leak unfortunately, it is a very novel and special topic, a fascinating problem. Contributions from NLP community and specialists will definitely bring PM into a bright future.

## Methods

### Data

The data comes from [Kaggle](https://www.kaggle.com/c/msk-redefining-cancer-treatment/overview) competition and Memorial Sloan Kettering Cancer Center.

Two data files were used in this project. There are 3321 data points in total. `Gene` and`Variation` are categorical.  `Class` contains 9 unique values.`TEXT` contains biomedical literature related to each class/ effect of genetic variants.

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

Here is the EDA demo and some findings. 

### Feature Extraction 

Methods:

* One-hot encoding gene and variation features

* Bag-of-Words (Tools: Count vectorizer and TFIDF vectorizer) and Word Embeddings (Source:  [stanford GloVe](https://nlp.stanford.edu/projects/glove/), [BioWordVec](https://github.com/ncbi-nlp/BioWordVec), [BioSentVec](https://github.com/ncbi-nlp/BioSentVec), [BioConceptVec](https://github.com/ncbi/BioConceptVec), [pubmed2018_w2v_200D](https://ia803102.us.archive.org/13/items/pubmed2018_w2v_200D.tar/README.txt), [pubmed2018_w2v_400D](https://github.com/RaRe-Technologies/gensim-data/issues/28)) for text feature.
* A pre-trained model trained on PubMed biomedicine literature by [Microsoft](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext#) for BERT model.

Features for training machine learning models: one-hot gene and variation + SVD truncated count vector and TFIDF vector of text. 

Features for training neural networks: feature vectors transformed by `pubmed2018_w2v_400D` pre-trained model (Except for BERT).

Here is an evaluation of some text feature extraction methods.

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

Here is an evaluation of model performance.

### Neural Networks (TODO)

pytorch - CNN, BiLSTM, RCNN, RNN+Attention, transformer ... describe them (refs)

which static word embedding were considered: Biowordvec_200D, pubmed_w2v_400D, Bioconcept_glove_100D  (refs) why-<u>link</u>

### Evaluation Metrics

* Log loss / Cross entropy loss (suggested by Kaggle competition)
* Balanced accuracy
* F1-score (micro-average)

## Results and Discussions

* An EDA demo
* Comparison of over sampling techniques

* Comparison of pre-trained word embedding models

* Evaluation of eight machine learning methods
* Evaluation of several neural networks

Please see this thesis for more discussions.

## Future Work

1. Combining NN models
2. Non-static word embedding models
3. Concatenating multiple word vector representations (with BioConceptVec)
4. Deal with imbalanced text data

## Environment

python 3.8

pytorch

## Directory

```
​```
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
│   │   └── CNN.py
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
```



## References

Transformer https://arxiv.org/pdf/1706.03762.pdf

BERT https://arxiv.org/pdf/1810.04805.pdf

TextCNN https://arxiv.org/pdf/1408.5882.pdf

BiLSTM https://www.sciencedirect.com/science/article/abs/pii/S0893608005001206

