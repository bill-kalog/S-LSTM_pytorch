import torch
import torch.nn

from torchtext import data
from torchtext import datasets
from utils import load_data
from sst_sent import SST_SENT

from IPython import embed


# train, dev, test, inputs, answers = load_data(SST_SENT, 'SST_FINE_PHRASES')
# train, dev, test, inputs, answers = load_data(SST_SENT, 'SST_FINE')
train, dev, test, inputs, answers = load_data(SST_SENT, 'SST_BIN')
# train, dev, test, inputs, answers = load_data(SST_SENT, 'SST_BIN_PHRASES')
embed()