import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os


from torchtext import data
from utils import load_data
from sst_sent import SST_SENT
from recurrent_models import RNN_encoder
from slstm import SLSTM

from IPython import embed

from tensorboardX import SummaryWriter


def do_forward_pass(batch, s_encoder, loss_function):
    l_probs, h_l, attention_weights = s_encoder(
        batch.text[0], batch.text[1])
    # l_probs, h_l, attention_weights = s_encoder(
        # batch.text[0])

    # calculate loss
    loss = loss_function(l_probs, batch.label - 1)
    # calculate accuracy
    _, predictions = torch.max(l_probs.data, 1)
    k_ = config['k_']
    topk_classes = torch.topk(l_probs.data, k_)[1] + 1
    filter_ = torch.eq(topk_classes, batch.label.data.unsqueeze(1))
    # embed()
    acc = float(torch.sum(torch.eq(predictions, batch.label.data - 1))) \
        / predictions.size()[0]
    acc_k = float(torch.sum(filter_)) / predictions.size()[0]
    return acc, loss, h_l, acc_k

# embed()
writer = SummaryWriter()
writer_path = list(writer.all_writers.keys())[0]
best_model_path = os.path.join(writer_path, 'best_dev_model')
if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)


# train, dev, test, inputs, answers = load_data(SST_SENT, 'SST_FINE_PHRASES')
# train, dev, test, inputs, answers = load_data(SST_SENT, 'SST_FINE')
train, dev, test, inputs, answers = load_data(SST_SENT, 'SST_BIN')
# train, dev, test, inputs, answers = load_data(SST_SENT, 'SST_BIN_PHRASES')

if torch.cuda.is_available():
    device_ = 0
else:
    device_ = -1

# load data and word embeddings
train_iter, dev_iter, test_iter = data.BucketIterator.splits(
    (train, dev, test),
    batch_sizes=(100, len(dev.examples), len(test.examples)),
    sort_key=lambda x: len(x.text), device=device_, sort_within_batch=True)


train_iter.init_epoch()
dev_iter.init_epoch()
test_iter.init_epoch()


# dnn_encoder = RNN_encoder(
#     input_dim=inputs.vocab.vectors.size()[1],
#     output_dim=100,
#     num_classes=len(answers.vocab.freqs.keys()),
#    vocab=inputs.vocab)

dnn_encoder = SLSTM(
    input_dim=inputs.vocab.vectors.size()[1],
    hidden_size=100,
    num_layers=1,
    window=2,
    num_classes=len(answers.vocab.freqs.keys()),
    vocab=inputs.vocab)

if torch.cuda.is_available():
    dnn_encoder.cuda(0)

# define loss funtion and optimizer
loss_function = nn.NLLLoss()
optimizer = optim.Adam(dnn_encoder.parameters())
# optimizer = optim.Adam(dnn_encoder.parameters(), weight_decay=1e-5)

save_graph_of_model = True
train_losses_list = []
train_acc_list = []
dev_acc_list = []
dev_losses_list = []
dev_max_acc = 0
max_acc_k = 0
acc_step = 0
best_model = None
config = {}
config['epochs'] = 5
config['k_'] = 2
wd = 1e-3  # weight decay
print('Starting training procedure')
# start training procedure
for batch_idx, batch in enumerate(train_iter):
    # embed()
    # switch to to training mode,zero out gradients
    dnn_encoder.train()
    optimizer.zero_grad()

    acc, loss, _, acc_k = do_forward_pass(
        batch, dnn_encoder, loss_function)

    loss.backward()
    # embed()
    # weight regularization
    for group in optimizer.param_groups:
        for param in group['params']:
            param.data = param.data.add(-wd * group['lr'], param.data)

    optimizer.step()
    train_acc_list.append(acc)
    train_losses_list.append(float(loss))
    # embed()
    writer.add_scalar('train/Loss', float(loss), batch_idx)
    writer.add_scalar('train/Acc', acc, batch_idx)

    writer.add_scalar('timers_train/bf_after', float(dnn_encoder.bef_aft_time_el), batch_idx)
    writer.add_scalar('timers_train/sent_time_el', float(dnn_encoder.sent_time_el), batch_idx)
    writer.add_scalar('timers_train/words_time_gates_el', float(dnn_encoder.words_time_gates_el), batch_idx)
    writer.add_scalar('timers_train/words_time_rest_el', float(dnn_encoder.words_time_rest_el), batch_idx)
    
    # evaluate on dev set
    # with torch.no_grad():
    dnn_encoder.eval()
    dev_batch = next(iter(dev_iter))
    # acc, loss, _ = perform_forward_pass(dev_batch, dnn_model, loss_function)
    acc, loss, _, acc_k = do_forward_pass(
        dev_batch, dnn_encoder, loss_function)
    dev_acc_list.append(acc)
    dev_losses_list.append(float(loss))

    writer.add_scalar('timers_dev/bf_after', float(dnn_encoder.bef_aft_time_el), batch_idx)
    writer.add_scalar('timers_dev/sent_time_el', float(dnn_encoder.sent_time_el), batch_idx)
    writer.add_scalar('timers_dev/words_time_gates_el', float(dnn_encoder.words_time_gates_el), batch_idx)
    writer.add_scalar('timers_dev/words_time_rest_el', float(dnn_encoder.words_time_rest_el), batch_idx)

    if acc > dev_max_acc:
        dev_max_acc = acc
        best_model = copy.deepcopy(dnn_encoder)
        max_acc_k = acc_k
        acc_step = batch_idx
        # check acc on test set
        test_batch = next(iter(test_iter))
        t_acc, t_loss, _, acc_k = do_forward_pass(test_batch, dnn_encoder, loss_function)
        print('accuraccy on test set is {} and at topk {}'.format(
            t_acc, acc_k))

    writer.add_scalar('dev/Loss', float(loss), batch_idx)
    writer.add_scalar('dev/Acc', acc, batch_idx)

    # getting tracing of legacy functions not supported error
    if save_graph_of_model:
        # with SummaryWriter(comment='Net') as w:
            # w.add_graph(dnn_encoder, (dev_batch.text[0], ))
        # writer.add_graph(dnn_encoder, batch.text[0])
        # l_probs, h_l, attention_weights = dnn_encoder(batch.text[0])
        # embed()
        # error check
        # https://github.com/lanpa/tensorboard-pytorch/pull/106
        # writer.add_graph(dnn_encoder, (batch.text[0], batch.text[1]), verbose=True)
        # torch.onnx.export(dnn_encoder, (batch.text[0], batch.text[1]), "./IndexLayer.pb", verbose=True)
        save_graph_of_model = False

    # info and stop criteria
    if train_iter.iterations % 100 == 0:
        print('epoch {} iteration {} current train acc {} train loss {} max dev acc {} at k {} at step {} \n'.format(
            train_iter.epoch, batch_idx, train_acc_list[-1],
            train_losses_list[-1], dev_max_acc, max_acc_k, acc_step))

    if train_iter.epoch > config['epochs']:
        break
writer.close()
embed()
