import torch
import torrch.nn as nn
import torch.nn.functional as F
from IPython import embed


class SLSTM(nn.Module):
    """docstring for SLSTM"""

    def __init__(self, input_dim, hidden_size, num_layers, num_classes, vocab=None):
        super(SLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.init_weights()

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
        self.W_i_hat_c = torch.Tensor(hidden_size, hidden_size)
        self.W_i_hat_lr = torch.Tensor(hidden_size, hidden_size)
        self.W_i_hat_x = torch.Tensor(hidden_size, hidden_size)
        self.W_i_hat_g = torch.Tensor(hidden_size, hidden_size)

        # self.l_hat
        self.W_l_hat_c = torch.Tensor(hidden_size, hidden_size)
        self.W_l_hat_lr = torch.Tensor(2 * hidden_size, hidden_size)
        self.W_l_hat_x = torch.Tensor(hidden_size, hidden_size)
        self.W_l_hat_g = torch.Tensor(hidden_size, hidden_size)

        # self.r_hat
        self.W_r_hat_c = torch.Tensor(hidden_size, hidden_size)
        self.W_r_hat_lr = torch.Tensor(2 * hidden_size, hidden_size)
        self.W_r_hat_x = torch.Tensor(hidden_size, hidden_size)
        self.W_r_hat_g = torch.Tensor(hidden_size, hidden_size)

        # self.f_hat
        self.W_f_hat_c = torch.Tensor(hidden_size, hidden_size)
        self.W_f_hat_lr = torch.Tensor(2 * hidden_size, hidden_size)
        self.W_f_hat_x = torch.Tensor(hidden_size, hidden_size)
        self.W_f_hat_g = torch.Tensor(hidden_size, hidden_size)

        # self.s_hat
        self.W_s_hat_c = torch.Tensor(hidden_size, hidden_size)
        self.W_s_hat_lr = torch.Tensor(2 * hidden_size, hidden_size)
        self.W_s_hat_x = torch.Tensor(hidden_size, hidden_size)
        self.W_s_hat_g = torch.Tensor(hidden_size, hidden_size)

        # self.output_gate
        self.W_output_gate_c = torch.Tensor(hidden_size, hidden_size)
        self.W_output_gate_lr = torch.Tensor(2 * hidden_size, hidden_size)
        self.W_output_gate_x = torch.Tensor(hidden_size, hidden_size)
        self.W_output_gate_g = torch.Tensor(hidden_size, hidden_size)

        # self.u_gate
        self.W_u_gate_c = torch.Tensor(hidden_size, hidden_size)
        self.W_u_gate_lr = torch.Tensor(2 * hidden_size, hidden_size)
        self.W_u_gate_x = torch.Tensor(hidden_size, hidden_size)
        self.W_u_gate_g = torch.Tensor(hidden_size, hidden_size)

        # biases
        self.i_hat_bias = torch.Tensor(hidden_size)
        self.l_hat_bias = torch.Tensor(hidden_size)
        self.r_hat_bias = torch.Tensor(hidden_size)
        self.f_hat_bias = torch.Tensor(hidden_size)
        self.s_hat_bias = torch.Tensor(hidden_size)
        self.output_gate_bias = torch.Tensor(hidden_size)
        self.u_gate_bias = torch.Tensor(hidden_size)

        # sentence vector weights

        # self.f_hat_g
        self.W_f_hat_g_g = torch.Tensor(hidden_size, hidden_size)
        self.W_f_hat_g_h = torch.Tensor(hidden_size, hidden_size)

        # self.f_hat_g_i
        self.W_f_hat_g_i_g = torch.Tensor(hidden_size, hidden_size)
        self.W_f_hat_g_i_h = torch.Tensor(hidden_size, hidden_size)

        # self.output_gate_g
        self.W_output_gate_g_g = torch.Tensor(hidden_size, hidden_size)
        self.W_output_gate_g_h = torch.Tensor(hidden_size, hidden_size)

        # biases sentence g
        self.f_hat_g_bias = torch.Tensor(hidden_size)
        self.f_hat_g_i_bias = torch.Tensor(hidden_size)
        self.output_gate_g = torch.Tensor(hidden_size)

        sigmoid = nn.Sigmoid()


    def forward():
        pass

    def init_weights():
        pass
    
    def slstm_step():
        
        self.i_hat = F.sigmoid()
        self.l_hat = F.sigmoid()
        self.r_hat = F.sigmoid()
        self.f_hat = F.sigmoid()
        self.s_hat = F.sigmoid()
        self.output_gate = F.sigmoid()
        self.u_gate = F.tanh()

        # sentence

        # average all hidden states

        # self.f_hat_g
        # self.f_hat_g_i
        # self.output_gate_g

    def get_hidden_states_before():
        pass

    def get_hidden_states_after():
        pass