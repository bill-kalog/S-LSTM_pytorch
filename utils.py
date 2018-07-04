from torchtext import data
from torchtext import datasets
from IPython import embed


def load_data(chosen_dataset, chosen_dataset_name):
    '''
    load data, split dataset and create vocabulary
    '''
    inputs = data.Field(
        lower=True, include_lengths=True, batch_first=True, tokenize='spacy')
    answers = data.Field(
        sequential=False)

    if chosen_dataset_name == 'SST_FINE_PHRASES':
        train, dev, test = chosen_dataset.splits(inputs, answers, fine_grained=True, use_phrases=True)
    elif chosen_dataset_name == 'SST_FINE':
        train, dev, test = chosen_dataset.splits(inputs, answers, fine_grained=True, use_phrases=False)
    elif chosen_dataset_name == 'SST_BIN':
        train, dev, test = chosen_dataset.splits(inputs, answers, fine_grained=False, use_phrases=False)
    elif chosen_dataset_name == 'SST_BIN_PHRASES':
        train, dev, test = chosen_dataset.splits(inputs, answers, fine_grained=False, use_phrases=True)
    else:
        raise NotImplementedError("Support for other datasets is currently missing!!")
    
    print('Building vocabulary')
    inputs.build_vocab(train, dev, test)
    inputs.vocab.load_vectors('glove.6B.100d')
    # inputs.vocab.load_vectors('glove.6B.50d')
    # inputs.vocab.load_vectors('glove.6B.300d')

    answers.build_vocab(train)
    return train, dev, test, inputs, answers