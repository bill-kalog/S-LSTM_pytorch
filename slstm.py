import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

if torch.cuda.is_available():  # gpu support
    dtype_float = torch.cuda.FloatTensor
    dtype_long = torch.cuda.LongTensor
else:
    # for cpu support
    dtype_float = torch.FloatTensor
    dtype_long = torch.LongTensor


class SLSTM(nn.Module):
    """docstring for SLSTM"""

    def __init__(
            self, input_dim, hidden_size, num_layers, window,
            num_classes, vocab=None):

        super(SLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.window = window
        self.num_classes = num_classes
        # initialize word vectors

        self.embed = nn.Embedding(len(vocab), input_dim).type(dtype_float)
        self.embed.weight.data.copy_(vocab.vectors)

        # declare weights
        # c : current
        # lr: left right weights
        # x : word weights
        # g : sentence weights
        # word processing weights
        # hidden_size = 10
        # torch.Tensor(hidden_size, hidden_size)
        # nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        # torch.normal(mean=0, std=torch.Tensor(hidden_size, hidden_size))
        #  m[self.somehting whatever].weight.data.normal_(0.0, 0.02)
        # https://gist.github.com/ikhlestov/031e0f4e83b968cede8df1d19f3d4714

        # self.i_hat
        self.W_i_hat_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_i_hat_c.data.normal_(0, 0.1)
        self.W_i_hat_lr = nn.Parameter(torch.Tensor(2 * hidden_size, hidden_size))
        self.W_i_hat_lr.data.normal_(0, 0.1)
        self.W_i_hat_x = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_i_hat_x.data.normal_(0, 0.1)
        self.W_i_hat_g = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_i_hat_g.data.normal_(0, 0.1)

        # self.l_hat
        self.W_l_hat_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_l_hat_c.data.normal_(0, 0.1)
        self.W_l_hat_lr = nn.Parameter(torch.Tensor(2 * hidden_size, hidden_size))
        self.W_l_hat_lr.data.normal_(0, 0.1)
        self.W_l_hat_x = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_l_hat_x.data.normal_(0, 0.1)
        self.W_l_hat_g = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_l_hat_g.data.normal_(0, 0.1)

        # self.r_hat
        self.W_r_hat_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_r_hat_c.data.normal_(0, 0.1)
        self.W_r_hat_lr = nn.Parameter(torch.Tensor(2 * hidden_size, hidden_size))
        self.W_r_hat_lr.data.normal_(0, 0.1)
        self.W_r_hat_x = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_r_hat_x.data.normal_(0, 0.1)
        self.W_r_hat_g = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_r_hat_g.data.normal_(0, 0.1)

        # self.f_hat
        self.W_f_hat_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_f_hat_c.data.normal_(0, 0.1)
        self.W_f_hat_lr = nn.Parameter(torch.Tensor(2 * hidden_size, hidden_size))
        self.W_f_hat_lr.data.normal_(0, 0.1)
        self.W_f_hat_x = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_f_hat_x.data.normal_(0, 0.1)
        self.W_f_hat_g = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_f_hat_g.data.normal_(0, 0.1)

        # self.s_hat
        self.W_s_hat_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_s_hat_c.data.normal_(0, 0.1)
        self.W_s_hat_lr = nn.Parameter(torch.Tensor(2 * hidden_size, hidden_size))
        self.W_s_hat_lr.data.normal_(0, 0.1)
        self.W_s_hat_x = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_s_hat_x.data.normal_(0, 0.1)
        self.W_s_hat_g = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_s_hat_g.data.normal_(0, 0.1)

        # self.output_gate
        self.W_output_gate_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_output_gate_c.data.normal_(0, 0.1)
        self.W_output_gate_lr = nn.Parameter(torch.Tensor(2 * hidden_size, hidden_size))
        self.W_output_gate_lr.data.normal_(0, 0.1)
        self.W_output_gate_x = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_output_gate_x.data.normal_(0, 0.1)
        self.W_output_gate_g = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_output_gate_g.data.normal_(0, 0.1)

        # self.u_gate
        self.W_u_gate_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_u_gate_c.data.normal_(0, 0.1)
        self.W_u_gate_lr = nn.Parameter(torch.Tensor(2 * hidden_size, hidden_size))
        self.W_u_gate_lr.data.normal_(0, 0.1)
        self.W_u_gate_x = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_u_gate_x.data.normal_(0, 0.1)
        self.W_u_gate_g = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_u_gate_g.data.normal_(0, 0.1)

        # biases
        self.i_hat_bias = nn.Parameter(torch.Tensor(hidden_size))
        self.i_hat_bias.data.normal_(0, 0.1)
        self.l_hat_bias = nn.Parameter(torch.Tensor(hidden_size))
        self.l_hat_bias.data.normal_(0, 0.1)
        self.r_hat_bias = nn.Parameter(torch.Tensor(hidden_size))
        self.r_hat_bias.data.normal_(0, 0.1)
        self.f_hat_bias = nn.Parameter(torch.Tensor(hidden_size))
        self.f_hat_bias.data.normal_(0, 0.1)
        self.s_hat_bias = nn.Parameter(torch.Tensor(hidden_size))
        self.s_hat_bias.data.normal_(0, 0.1)
        self.output_gate_bias = nn.Parameter(torch.Tensor(hidden_size))
        self.output_gate_bias.data.normal_(0, 0.1)
        self.u_gate_bias = nn.Parameter(torch.Tensor(hidden_size))
        self.u_gate_bias.data.normal_(0, 0.1)

        # sentence vector weights

        # self.f_hat_g
        self.W_f_hat_g_g = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_f_hat_g_g.data.normal_(0, 0.1)
        self.W_f_hat_g_h = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_f_hat_g_h.data.normal_(0, 0.1)

        # self.f_hat_g_i
        self.W_f_hat_g_i_g = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_f_hat_g_i_g.data.normal_(0, 0.1)
        self.W_f_hat_g_i_h = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_f_hat_g_i_h.data.normal_(0, 0.1)

        # self.output_gate_g
        self.W_output_gate_g_g = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_output_gate_g_g.data.normal_(0, 0.1)
        self.W_output_gate_g_h = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.W_output_gate_g_h.data.normal_(0, 0.1)

        # biases sentence g
        self.f_hat_g_bias = nn.Parameter(torch.Tensor(hidden_size))
        self.f_hat_g_bias.data.normal_(0, 0.1)
        self.f_hat_g_i_bias = nn.Parameter(torch.Tensor(hidden_size))
        self.f_hat_g_i_bias.data.normal_(0, 0.1)
        self.output_gate_g_bias = nn.Parameter(torch.Tensor(hidden_size))
        self.output_gate_g_bias.data.normal_(0, 0.1)

        self.linear = nn.Linear(self.hidden_size, self.num_classes)
        self.w1 = torch.randn(hidden_size, hidden_size, requires_grad=True)
        self.a_2 = nn.Parameter(torch.ones(5))
        self.b_2 = nn.Parameter(torch.zeros(5))

        # self.init_weights()

    def forward(self, input_sentences, sentences_length):
        # embed()
        # print (" matrix : {}".format(self.W_output_gate_g_g))
        word_vectors = self.embed(input_sentences)
        # randomly initialize states
        # interval_ = [-0.05, 0.05]
        interval_ = 0.05 - - 0.05
        min_ = -0.05
        hh_w = min_ + torch.rand(word_vectors.shape) * interval_
        cs_w = min_ + torch.rand(word_vectors.shape) * interval_
        # embed()
        # get mean for the sentence representations
        sent_vec = hh_w.mean(dim=1)
        sent_cs = cs_w.mean(dim=1)
        for i in range(self.num_layers):
            hh_w, cs_w, sent_vec, sent_cs = \
                self.slstm_step(hh_w, cs_w, sent_vec, sent_cs, word_vectors)

        fc_out = self.linear(sent_vec)
        log_softmax = F.log_softmax(fc_out, dim=1)
        return log_softmax, sent_vec, None

    def slstm_step(self, hidden_states_words, cell_state_words,
                   sentence_vector, sentence_cell_state, word_vectors):

        averaged_word_hh = hidden_states_words.mean(dim=1)

        # calculate sentence node (dummy)
        f_hat_g_gate = F.sigmoid(
            sentence_vector @ self.W_f_hat_g_g +
            averaged_word_hh @ self.W_f_hat_g_h + self.f_hat_g_bias
        )

        output_gate_g = F.sigmoid(
            sentence_vector @ self.W_output_gate_g_g +
            averaged_word_hh @ self.W_output_gate_g_h + self.output_gate_bias
        )

        f_hat_g_i = F.sigmoid(
            torch.unsqueeze(sentence_vector @ self.W_f_hat_g_i_g, 1) +
            hidden_states_words @ self.W_f_hat_g_i_h +
            torch.unsqueeze(self.f_hat_g_i_bias, 0)
        )
        # softmax for each f_i in  f_hat_i
        g_softmax_scores = F.softmax(f_hat_g_i, 2)
        f_i_g_elwise_c_i = torch.sum(g_softmax_scores * cell_state_words, 1)
        # calculate new cell state
        new_sentence_cell_state = F.softmax(f_hat_g_gate, 1) * \
            sentence_cell_state + f_i_g_elwise_c_i

        new_sentence_vector = output_gate_g * F.tanh(new_sentence_cell_state)

        # update word node states

        # get before and after words i.e ξ_i
        hidden_states_before = [self.get_hidden_states_before(
            hidden_states_words, step + 1) for step in range(self.window)]
        # sum the tensors of the different windows together
        hidden_states_before = self.sum_together(hidden_states_before)
        hidden_states_after = [self.get_hidden_states_after(
            hidden_states_words, step + 1) for step in range(self.window)]
        hidden_states_after = self.sum_together(hidden_states_after)

        # do the same for the cell states
        cell_state_before = [self.get_hidden_states_before(
            cell_state_words, step + 1) for step in range(self.window)]
        cell_state_before = self.sum_together(cell_state_before)
        cell_state_after = [self.get_hidden_states_after(
            cell_state_words, step + 1) for step in range(self.window)]
        cell_state_after = self.sum_together(cell_state_after)

        # coancatenate states
        concat_before_after = torch.cat(
            (hidden_states_before, hidden_states_after), 2)

        # calculate/update word states
        i_hat = F.sigmoid(
            concat_before_after @ self. W_i_hat_lr +
            hidden_states_words @ self.W_i_hat_c +
            word_vectors @ self.W_i_hat_x +
            torch.unsqueeze(new_sentence_vector @ self.W_i_hat_g, 1) +
            torch.unsqueeze(self.i_hat_bias, 0)
        )

        l_hat = F.sigmoid(
            concat_before_after @ self.W_l_hat_lr +
            hidden_states_words @ self.W_l_hat_c +
            word_vectors @ self.W_l_hat_x +
            torch.unsqueeze(new_sentence_vector @ self.W_l_hat_g, 1) +
            torch.unsqueeze(self.l_hat_bias, 0)
        )

        r_hat = F.sigmoid(
            concat_before_after @ self.W_r_hat_lr +
            hidden_states_words @ self.W_r_hat_c +
            word_vectors @ self.W_r_hat_x +
            torch.unsqueeze(new_sentence_vector @ self.W_r_hat_g, 1) +
            torch.unsqueeze(self.r_hat_bias, 0)
        )

        f_hat = F.sigmoid(
            concat_before_after @ self.W_f_hat_lr +
            hidden_states_words @ self.W_f_hat_c +
            word_vectors @ self.W_f_hat_x +
            torch.unsqueeze(new_sentence_vector @ self.W_f_hat_g, 1) +
            torch.unsqueeze(self.f_hat_bias, 0)
        )

        s_hat = F.sigmoid(
            concat_before_after @ self.W_s_hat_lr +
            hidden_states_words @ self.W_s_hat_c +
            word_vectors @ self.W_s_hat_x +
            torch.unsqueeze(new_sentence_vector @ self.W_s_hat_g, 1) +
            torch.unsqueeze(self.s_hat_bias, 0)
        )

        output_gate = F.sigmoid(
            concat_before_after @ self.W_output_gate_lr +
            hidden_states_words @ self.W_output_gate_c +
            word_vectors @ self.W_output_gate_x +
            torch.unsqueeze(new_sentence_vector @ self.W_output_gate_g, 1) +
            torch.unsqueeze(self.output_gate_bias, 0)
        )

        u_gate = F.tanh(
            concat_before_after @ self.W_u_gate_lr +
            hidden_states_words @ self.W_u_gate_c +
            word_vectors @ self.W_u_gate_x +
            torch.unsqueeze(new_sentence_vector @ self.W_u_gate_g, 1) +
            torch.unsqueeze(self.u_gate_bias, 0)
        )

        conc_gates = torch.cat([i_hat, l_hat, r_hat, f_hat, s_hat], 2)
        conc_gates = F.softmax(conc_gates, 2)

        i_t, l_t, r_t, f_t, s_t = torch.split(conc_gates, self.hidden_size, 2)

        # the caclulation from tf doen'st have a u_gate as in the paper
        new_cell_t_words = l_t * cell_state_before + f_t * cell_state_words + \
            r_t * cell_state_after + \
            s_t * torch.unsqueeze(sentence_cell_state, 1) + \
            i_t * u_gate

        new_hidden_state_words = output_gate * F.tanh(new_cell_t_words)

        return new_hidden_state_words, new_cell_t_words, \
            new_sentence_vector, new_sentence_cell_state

    def get_hidden_states_before(self, hidden_states, step):
        padding = torch.zeros(
            [hidden_states.shape[0], step, hidden_states.shape[2]])
        displaced_hidden_states = hidden_states[:, :-step, :]
        return torch.cat((padding, displaced_hidden_states), 1)

    def get_hidden_states_after(self, hidden_states, step):
        padding = torch.zeros(
            [hidden_states.shape[0], step, hidden_states.shape[2]])
        displaced_hidden_states = hidden_states[:, step:, :]
        return torch.cat((displaced_hidden_states, padding), 1)

    def sum_together(self, tensor_list):
        combined_tensors = None
        for tensor_ in tensor_list:
            if combined_tensors is None:
                combined_tensors = tensor_
            else:
                combined_tensors = combined_tensors + tensor_
        return combined_tensors

    def init_weights(self, model):
        for param in model.parameters():
            param.requires_grad = True

        pass
