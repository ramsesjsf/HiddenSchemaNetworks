import io
import logging
import os
from collections import defaultdict
import numpy as np

from tqdm import tqdm
from torchtext.utils import download_from_url, extract_archive
from transformers import GPT2Tokenizer, BertTokenizer
import pickle
import torch
from torch.utils.data import Dataset
from collections import namedtuple
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from torchtext.data.functional import numericalize_tokens_from_iterator
from hiddenschemanetworks.data.utils import build_vocab_from_iterator





URLS = {
    'PennTreebank':
        ['https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt',
         'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt',
         'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt'],
    'YahooAnswers':
        ['https://raw.githubusercontent.com/fangleai/Implicit-LVM/master/lang_model_yahoo/data/train.txt',
         'https://raw.githubusercontent.com/fangleai/Implicit-LVM/master/lang_model_yahoo/data/test.txt',
         'https://raw.githubusercontent.com/fangleai/Implicit-LVM/master/lang_model_yahoo/data/val.txt'],
    'YelpReview':
        ['https://github.com/fangleai/Implicit-LVM/raw/master/lang_model_yelp/data/yelp.train.txt',
         'https://github.com/fangleai/Implicit-LVM/raw/master/lang_model_yelp/data/yelp.test.txt',
         'https://github.com/fangleai/Implicit-LVM/raw/master/lang_model_yelp/data/yelp.valid.txt']
}

def _get_datafile_path(key, extracted_files):
    for fname in extracted_files:
        if key in fname:
            return fname

class LanguageModelingDatasetPretrained(torch.utils.data.Dataset):
    """
    Defines a dataset for language modeling using pretrained tokenizers from huggingface transformers.
    """

    def __init__(self, data, tokenizers):
        """
        Initiate language modeling dataset using pretrained tokenizers from huggingface.
        """

        super(LanguageModelingDatasetPretrained, self).__init__()
        self.data = data
        self.tokenizer_enc, self.tokenizer_dec = tokenizers
        self.pad_token_id = -100

    def __getitem__(self, i):
        minibatch = {'input_enc': np.asarray(self.data[i]['input_enc'], dtype=np.int64),
                     'attn_mask_enc': np.asarray(self.data[i]['attn_mask_enc']),
                     'length_enc': np.asarray(self.data[i]['length_enc']),
                     'input_dec': np.asarray(self.data[i]['input_dec'], dtype=np.int64),
                     'target_dec': np.asarray(self.data[i]['target_dec'], dtype=np.int64),
                     'attn_mask_dec': np.asarray(self.data[i]['attn_mask_dec']),
                     'length_dec': np.asarray(self.data[i]['length_dec'])
                     }
        # yahoo has labels
        if 'label' in self.data[i].keys():
            minibatch.update({'label': np.asarray(self.data[i][0]['label'])})
        return minibatch

    def __iter__(self):
        for i in range(self.__len__()):
            yield self[i]

    def __len__(self):
        return len(self.data)

    def get_pad_token_id(self):
        return self.pad_token_id

    def reverse(self, batch):
        batch[batch == self.pad_token_id] = self.tokenizer_dec.eos_token_id
        sentences = self.tokenizer_dec.batch_decode(batch, skip_special_tokens=True)
        return sentences

# add label at beginning of each line of yahoo
def preprocess_yahoo(file):
    src = open(file, "rt")
    label = -1
    file_iter = [row for row in src]
    data = ''
    for i, line in enumerate(file_iter):
        if i % (len(file_iter) / 10) == 0:
            label += 1
        data += str(label) + '\t' + line
    src.close()
    dest = open(file, "wt")
    dest.write(data)
    dest.close()

def _setup_datasets(dataset_name,
                    fix_len,
                    root='./data',
                    data_select=('train', 'test', 'valid'), ):

    if isinstance(data_select, str):
        data_select = [data_select]
    if not set(data_select).issubset({'train', 'test', 'valid'}):
        raise TypeError('data_select is not supported!')

    # get the pretrained tokenizers
    tokenizer_enc = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer_dec = GPT2Tokenizer.from_pretrained('gpt2')



    if dataset_name == 'PennTreebank':
        extracted_files = []
        select_to_index = {'train': 0, 'test': 1, 'valid': 2}
        for key in data_select:
            url_ = URLS['PennTreebank'][select_to_index[key]]
            _, filename = os.path.split(url_)
            path_ = os.path.join(root, filename)
            if os.path.exists(path_):
                extracted_files.append(path_)
            else:
                extracted_files.append(download_from_url(url_, root=root))

    elif dataset_name in ['YahooAnswers', 'YelpReview']:
        extracted_files = []
        select_to_index = {'train': 0, 'test': 1, 'valid': 2}
        for key in data_select:
            url_ = URLS[dataset_name][select_to_index[key]]
            _, filename = os.path.split(url_)
            path_ = os.path.join(root, filename)
            if not os.path.exists(path_.replace('val', 'valid')):
                path_ = download_from_url(url_, root=root)
                if dataset_name == 'YahooAnswers':
                    preprocess_yahoo(path_)
                os.rename(path_, path_.replace('val', 'valid'))
            extracted_files.append(path_.replace('val', 'valid'))
    else:
        extracted_files = []
        for key in data_select:

            file_ = os.path.join(root, f'{key}.txt')
            if not os.path.exists(file_):
                raise FileExistsError(f'File cannot be found at location {file_}')
            extracted_files.append(file_)

    _path = {}
    for item in data_select:
        _path[item] = _get_datafile_path(item, extracted_files)

    data = dict()

    for item in _path.keys():
        if item not in data_select:
            continue

        preprocessed_path = os.path.join(root, f'preprocessed_len{fix_len}_{item}.pkl')
        if os.path.exists(preprocessed_path):
            with open(preprocessed_path, 'rb') as file:
                data[item] = pickle.load(file)
            print(f'loading {item} data from {preprocessed_path}')
            continue

        data_set = defaultdict(dict)
        logging.info('Creating {} data'.format(item))
        _iter = iter(row for row in io.open(_path[item], encoding="utf8"))
        id = 0
        for row in tqdm(_iter, unit='data point', desc=f'Preparing {item} dataset'):
            row = row[:-1]  # remove \n at the end of each line
            ### data set specific alterations ###
            if dataset_name in ['YahooAnswersPretrained', 'YelpReviewPretrained']:
                data_set[id]['label'] = int(row[0])
                row = row[2:]  # remove the label (and the space after it)

            ### BERT tokenizer ###

            tokens_attns = tokenizer_enc(row,
                                     truncation=True,
                                     max_length=fix_len,
                                     return_length=True,
                                     add_special_tokens=True)

            pad_len = fix_len - tokens_attns['length']

            data_set[id]['length_enc'] = tokens_attns['length'] + 1
            data_set[id]['input_enc'] = tokens_attns['input_ids'] + [tokenizer_enc.pad_token_id] * pad_len
            data_set[id]['attn_mask_enc'] = tokens_attns['attention_mask'] + [0] * pad_len


            ### GPT2 tokenizer ###
            tokens_attns = tokenizer_dec(row,
                                     truncation=True,
                                     max_length=fix_len-1,
                                     return_length=True)


            pad_len = fix_len - tokens_attns['length'] - 1

            data_set[id]['length_dec'] = tokens_attns['length'] + 1
            data_set[id]['input_dec'] = [tokenizer_dec.bos_token_id] + tokens_attns['input_ids'] + \
                                          [tokenizer_dec.eos_token_id] * pad_len
            data_set[id]['target_dec'] = tokens_attns['input_ids'] + [tokenizer_dec.eos_token_id] + \
                                           [-100] * pad_len
            data_set[id]['attn_mask_dec'] = tokens_attns['attention_mask'] + [1] + [0] * pad_len

            assert len(data_set[id]['input_dec']) == len(data_set[id]['target_dec']) \
                   == len(data_set[id]['attn_mask_dec']) == fix_len

            id += 1
        data[item] = data_set
        # save data dict to disk
        with open(preprocessed_path, 'wb+') as file:
            pickle.dump(dict(data_set), file)
    for key in data_select:
        if not data[key]:
            raise TypeError('Dataset {} is empty!'.format(key))


    return tuple(LanguageModelingDatasetPretrained(data[d], (tokenizer_enc, tokenizer_dec)) for d in data_select)


def PennTreebank(*args, **kwargs):
    """ Defines PennTreebank datasets.

    Create language modeling dataset: PennTreebank
    Separately returns the train/test/valid set

    Arguments:
        pretrained_tokenizer: list of strings from {'GPT2', 'BERT'} that correspond to the tokenizers of huggingface
            transformers {'gpt2', 'bert-base-uncased'}
        root: Directory where the datasets are saved. Default: ".data"
        data_select: a string or tupel for the returned datasets
            (Default: ('train', 'test','valid'))
            By default, all the three datasets (train, test, valid) are generated. Users
            could also choose any one or two of them, for example ('train', 'test') or
            just a string 'train'. If 'train' is not in the tuple or string, a vocab
            object should be provided which will be used to process valid and/or test
            data.
    """

    return _setup_datasets(*(("PennTreebank",) + args), **kwargs)


def YahooAnswers(*args, **kwargs):
    return _setup_datasets(*(("YahooAnswers",) + args), **kwargs)

def YelpReview(*args, **kwargs):
    return _setup_datasets(*(("YelpReview",) + args), **kwargs)

Point = namedtuple('Point', 'walks, text')

class SyntheticSchemataDataset(Dataset):
    data: dict
    vocab: dict
    time: dict

    def __init__(self, path_to_data: str, type: str, schemata_type: str):
        super(SyntheticSchemataDataset, self).__init__()
        assert os.path.exists(path_to_data)
        self.walks = torch.Tensor(pickle.load(open(os.path.join(path_to_data, "{0}.{1}_walks.pkl".format(schemata_type,type)), "rb")))
        self.text = torch.Tensor(pickle.load(open(os.path.join(path_to_data, "{0}_text_numericalized.pkl".format(type)), "rb")))

    def __getitem__(self, i):
        return Point(self.walks[i],self.text[i])

    def __len__(self):
        return len(self.walks)

    def __iter__(self):
        for i in range(self.__len__()):
            yield Point(self.walks[i],self.text[i])

    def get_vocab(self):
        return self.vocab

