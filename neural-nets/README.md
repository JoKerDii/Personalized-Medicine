# Text Classification



## Environment 

python 3.8  

pytorch 1.7.0

keras 2.4.3 (needed when preprocess the data)

sklearn 

## Training

```shell
# Training and Testing a model
python run.py --model=CNN 
python run.py --model=BiLSTM

# Choose device. Default number is 'cpu'. Can also be 'gpu'
python run.py --model=CNN --device='gpu'

# Specify cuda number. Default number is 0
python run.py --model=CNN --device='gpu' --cuda=3

# Whether use visdom for visualization. Default is False. Visdom requires pre-initiated visdom.server: https://github.com/facebookresearch/visdom
python run.py --model=CNN --device='gpu' --cuda=3 --visdom=True
```

## Result

Models|Balanced Accuracy (valid)|F1 score (valid)
--|--|--
CNN|60.51%|0.6329
BiLSTM|65.34%|0.6211 
RNN_Attention||
RCNN||
BERT||


## References
[1] Convolutional Neural Networks for Sentence Classification  
[2] Recurrent Neural Network for Text Classification with Multi-Task Learning  
[3] Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification  
[4] Recurrent Convolutional Neural Networks for Text Classification  
[5] Bag of Tricks for Efficient Text Classification  
[6] Deep Pyramid Convolutional Neural Networks for Text Categorization  
[7] Attention Is All You Need  
