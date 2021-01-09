# word-embedding-and-bow



**Comparison:**



|                                                              | with  dummy variables ||| without  dummy variables ||| without  dummy variables and common words |||
| ------------------------------------------------------------ | --------------------- | ------------------ | --------- | ------------------------ | ------------------ | --------- | ----------------------------------------- | ------------------ | --------- |
| Methods\ Performance on testing data                         | Log  Loss             | Balanced  Accuracy | F1  Score | Log  Loss                | Balanced  Accuracy | F1  Score | Log  Loss                                 | Balanced  Accuracy | F1  Score |
| SVD + TFIDF                                                  | 1.364                 | 0.488              | 0.629     | 1.365                    | 0.477              | 0.620     | 1.264                                     | 0.494              | 0.620     |
| SVD + Count                                                  | 1.429                 | 0.427              | 0.593     | 1.409                    | 0.432              | 0.599     | 1.313                                     | 0.464              | 0.617     |
| SVD + TFIDF +  Count                                         | 1.393                 | 0.458              | 0.616     | 1.428                    | 0.462              | 0.622     | 1.285                                     | 0.478              | 0.621     |
| [glove 840B 300d](https://nlp.stanford.edu/projects/glove/) [1] | 1.192                 | 0.457              | 0.622     | 1.180                    | 0.466              | 0.618     | 1.171                                     | 0.459              | 0.619     |
| [pubmed2018 w2v 200D](https://github.com/RaRe-Technologies/gensim-data/issues/28) [2] | 1.180                 | 0.484              | 0.636     | 1.233                    | 0.477              | 0.623     | 1.176                                     | 0.481              | 0.621     |
| [pubmed2018 w2v 400D](https://github.com/RaRe-Technologies/gensim-data/issues/28) [2] | 1.149                 | 0.483              | 0.631     | 1.190                    | 0.480              | 0.625     | 1.219                                     | 0.487              | 0.627     |
| [BioWordVec 200D](https://github.com/ncbi-nlp/BioWordVec) [3] | 1.197                 | 0.477              | 0.627     | 1.192                    | 0.480              | 0.620     | 1.217                                     | 0.481              | 0.626     |
| [BioSentVec 700D](https://github.com/ncbi-nlp/BioSentVec) [4] | 1.377                 | 0.468              | 0.626     | 1.334                    | 0.465              | 0.623     | 1.301                                     | 0.470              | 0.624     |
| [BioConceptVec fasttext 100D](https://github.com/ncbi/BioConceptVec) [5] | 1.315                 | 0.457              | 0.608     | 1.368                    | 0.460              | 0.600     | 1.325                                     | 0.465              | 0.602     |
| [BioConceptVec word2vec cbow 100D](https://github.com/ncbi/BioConceptVec)[5] | 1.219                 | 0.465              | 0.626     | 1.342                    | 0.471              | 0.613     | 1.346                                     | 0.478              | 0.613     |
| [Bioconceptvec word2vec skipgram 100D](https://github.com/ncbi/BioConceptVec)[5] | 1.171                 | 0.479              | 0.629     | 1.187                    | 0.489              | 0.627     | 1.216                                     | 0.490              | 0.621     |
| [Bioconceptvec glove 100D](https://github.com/ncbi/BioConceptVec)[5] | 1.184                 | 0.472              | 0.625     | 1.258                    | 0.474              | 0.616     | 1.236                                     | 0.482              | 0.620     |



**References:**

[1] Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf). [[bib](https://nlp.stanford.edu/pubs/glove.bib)]

[2] R. McDonald, G. Brokos and I. Androutsopoulos, "Deep Relevance Ranking Using Enhanced Document-Query Interactions". Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP 2018), Brussels, Belgium, 2018.

[3] Zhang Y, Chen Q, Yang Z, Lin H, Lu Z. [BioWordVec, improving biomedical word embeddings with subword information and MeSH](https://www.nature.com/articles/s41597-019-0055-0). Scientific Data. 2019.

[4] Chen Q, Peng Y, Lu Z. [BioSentVec: creating sentence embeddings for biomedical texts.](http://arxiv.org/abs/1810.09302) The 7th IEEE International Conference on Healthcare Informatics. 2019.

[5] Chen, Q., Lee, K., Yan, S., Kim, S., Wei, C. H., & Lu, Z. (2019). [BioConceptVec: creating and evaluating literature-based biomedical concept embeddings on a large scale](https://arxiv.org/ftp/arxiv/papers/1912/1912.10846.pdf). To appear in PLOS Computational Biology.