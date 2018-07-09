# S-LSTM_pytorch
**Toy** implementation of the paper 
[Sentence-State LSTM for Text Representation](https://arxiv.org/pdf/1805.02474.pdf) 
using PyTorch (only for classification)
based on the tensorflow implemetation of the [author](https://github.com/leuchine/S-LSTM)

## structure
Right now there are two different models implemented.
The S-LSTM model and a vanilla bidirectional RNN as a baseline. In
addition, there is a simple attention module that can be put on top
of either model. 


## Datasets
At the moment only the ability of loading the [Stanford sentiment treebank](https://nlp.stanford.edu/sentiment/treebank.html) 
(all versions) via [torchtext](https://github.com/pytorch/text/tree/master/torchtext) is available. So, in general
adding support for any additional datasets should be straight forward
enough with torchtext

## tensorboard
One can use [tensorboard-pytorch](https://github.com/lanpa/tensorboard-pytorch) for plotting accuracies etc. More info can be found in this [blogpost](https://medium.com/@dexterhuang/tensorboard-for-pytorch-201a228533c5)

## issues
the implementation seems kinda slow at the moment and because I don't
have an access to a gpu right now I haven't benchmarked it there.
Also, I have done almost zero hyperparameter tuning. 