import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
from attention import WeightedSumAttention


if torch.cuda.is_available():  # gpu support
    dtype_float = torch.cuda.FloatTensor
    dtype_long = torch.cuda.LongTensor
else:
    # for cpu support
    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor


class RNN_encoder(nn.Module):
    """a bidirectional RNN encoder with attention

    Inputs:
        input_dim: dimensions of input vectors
        output_dim: dimensionality of hidden representation and
                    output vector
        num_classes: number of classes for the softamx layer
        n_layers: number of stacked reccurent layers
        dropout: percentage of droppout between recurent layers
        vocab: vocabulary with vector initializations
    """

    def __init__(self, input_dim, output_dim, num_classes,
                 n_layers=1, dropout=0.5, vocab=None):
        super(RNN_encoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.num_classes = num_classes

        self.embed = nn.Embedding(len(vocab), input_dim).type(dtype_float)
        self.embed.weight.data.copy_(vocab.vectors)
        self.encoder = nn.GRU(
            self.input_dim, self.output_dim, self.n_layers,
            dropout=self.dropout, bidirectional=True)
        self.calc_attention_values = WeightedSumAttention(self.output_dim * 2)
        self.linear = nn.Linear(self.output_dim * 2, self.num_classes)

    def forward(self, input_sentences, sentences_length, hidden_vectors=None):
        '''
        forward pass
        Inputs:
            input_sentences: batch of input sentence vectors
            sentences_length: list of sentence length in the bach
            hidden_vectors : initialization for hidden embedding if applicable
        Outputs:
            log_softamx: softmax values
            attented_representations: final sentence vectors
            attention_weights: attention distributions per sentence
        '''
        word_vectors = self.embed(input_sentences)
        # put seqeunce length as first dimension
        word_vectors_transposed = word_vectors.transpose(1, 0)
        # force weights to exist in a continuous chank of memory
        self.encoder.flatten_parameters()
        # Pack a Variable containing padded sequences of variable length.
        packed_vectors = torch.nn.utils.rnn.pack_padded_sequence(
            word_vectors_transposed, sentences_length.tolist())
        output, h_n = self.encoder(packed_vectors, hidden_vectors)
        # pad shorter outputs with zeros
        output, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(
            output)
        # get attention weights
        attention_weights = self.calc_attention_values(output)

        # do a weighted sum attention
        output = output.transpose(1, 0)
        attention_weights = attention_weights.transpose(1, 0).unsqueeze(1)
        attented_representations = attention_weights.bmm(output).squeeze(1)

        # pass output through a fc layer
        fc_out = self.linear(attented_representations)
        log_softmax = F.log_softmax(fc_out, dim=1)

        return log_softmax, attented_representations, attention_weights
