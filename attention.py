import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Attention modules
'''


class WeightedSumAttention(nn.Module):
    """weighted sumAttention"""

    def __init__(self, hidden_dim):
        super(WeightedSumAttention, self).__init__()
        self.hidden_dim = hidden_dim
        # use a single fc layer to get a score
        self.att_nn = nn.Linear(self.hidden_dim, 1)
        # self.linear2 = nn.Linear(self.hidden_dim, 200)
        # self.linear3 = nn.Linear(200, 1)

    def forward(self, encoder_output):
        # attention scores dim [max_len, batch_size, 1]
        attention_scores = self.att_nn(encoder_output)

        # put one more layer in the calculation
        # relu_1 = F.relu(self.linear2(encoder_output))
        # attention_scores = self.linear3(relu_1)

        # transform to a distribution. dim [max_len, batch_size, 1] -> [len, bsize]
        attention_distribution = F.softmax(attention_scores, 0).squeeze(2)

        return attention_distribution
