import io
import logging
import os
from collections import defaultdict
import numpy as np

from tqdm import tqdm
from torchtext.utils import download_from_url
from transformers import GPT2Tokenizer, BertTokenizer
import pickle
import torch
from torch.utils.data import Dataset
from collections import namedtuple
from torchtext.vocab import build_vocab_from_iterator, Vocab





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
         'https://github.com/fangleai/Implicit-LVM/raw/master/lang_model_yelp/data/yelp.valid.txt'],
    'PCFG':
        [('https://raw.githubusercontent.com/i-machine-think/am-i-compositional/master/data/pcfgset/pcfgset/train.src',
          'https://raw.githubusercontent.com/i-machine-think/am-i-compositional/master/data/pcfgset/pcfgset/train.tgt'),
         ('https://raw.githubusercontent.com/i-machine-think/am-i-compositional/master/data/pcfgset/pcfgset/test.src',
          'https://raw.githubusercontent.com/i-machine-think/am-i-compositional/master/data/pcfgset/pcfgset/test.tgt'),
         ('https://raw.githubusercontent.com/i-machine-think/am-i-compositional/master/data/pcfgset/pcfgset/dev.src',
          'https://raw.githubusercontent.com/i-machine-think/am-i-compositional/master/data/pcfgset/pcfgset/dev.tgt')]
}

def _get_datafile_path(key, extracted_files):
    for fname in extracted_files:
        if key in fname:
            return fname

class LanguageModelingDatasetPretrained(Dataset):
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

def _setup_pcfg(root='./data', atomic_style=False, ae_style=False, train_on_object_only=True):
    """
    if atomic_style is false, this will be a translation task, i.e. the decoder input will only be the target
    else, the decoder input is source + target

    if ae_style is false, the encoder input is only the source
    else, the encoder input is source + <gen token> + target

    if train_on_object_only is true, the decoder target is only the target
    else, the decoder target is source + target
    only valid if atomic_style is true

    the maximum length of inputs is 71
    the maximum length of targets is 201 (only 3 examples with more)
    if atomic_style, maximum lenght of targets is 250
    """
    fix_len_src = 235 if ae_style else 71
    fix_len_tgt = 235 if atomic_style else 201


    local_paths = []
    for url_tuple in URLS['PCFG']:
        path_tuple = []
        for url in url_tuple:
            filename = os.path.basename(url)
            path = os.path.join(root, filename)
            if not os.path.exists(path):
                download_from_url(url, root=root)
            path_tuple.append(path)
        local_paths.append(tuple(path_tuple))

    dataset_list = []
    if atomic_style:
        specials = ['<sos>', '<eos>', '<gen>']
    else:
        specials = ['<sos>', '<eos>']

    for i, path_tuple in enumerate(local_paths):
        source_path, target_path = path_tuple
        source_file = open(source_path, 'r').read()
        target_file = open(target_path, 'r').read()
        if i == 0:
            vocab = Vocab(build_vocab_from_iterator([source_file.split()], specials=specials))
        SOS = vocab['<sos>']
        EOS = vocab['<eos>']
        if atomic_style:
            GEN = vocab['<gen>']

        dataset = []
        for source_line, target_line in zip(source_file.split('\n')[:-1], target_file.split('\n')[:-1]):
            data = dict()

            source_tokens = source_line.split(' ')
            source_tokens = vocab(source_tokens[:min(len(source_tokens), fix_len_src)])
            target_tokens = target_line.split(' ')
            max_len_tgt = fix_len_tgt - len(source_tokens) - 2 if atomic_style else fix_len_tgt - 1
            target_tokens = vocab(target_tokens[:min(len(target_tokens), max_len_tgt)])

            source_len = len(source_tokens)
            target_len = len(target_tokens)
            source_pad_len = fix_len_src - source_len
            target_pad_len = fix_len_tgt - target_len - 1

            if ae_style:
                data['length_enc'] = source_len + target_len + 1
                total_pad_len = fix_len_src - source_len - target_len - 1
                data['input_enc'] = source_tokens + [GEN] + target_tokens + [EOS] * total_pad_len
                data['attn_mask_enc'] = [1] * (source_len + target_len + 1) + [0] * total_pad_len
                data['token_type_ids'] = [0] * source_len + [1] * source_pad_len
            else:
                data['length_enc'] = source_len
                data['input_enc'] = source_tokens + [EOS] * source_pad_len
                data['attn_mask_enc'] = [1] * source_len + [0] * source_pad_len

            if atomic_style:
                if train_on_object_only:
                    total_pad_len = fix_len_tgt - source_len - target_len - 2
                    data['length_dec'] = target_len + source_len + 2
                    data['input_dec'] = [SOS] + source_tokens + [GEN] + target_tokens + [EOS] * total_pad_len
                    data['attn_mask_dec'] = [0] * (source_len + 1) + [1] * (target_len + 1) + [0] * total_pad_len
                    data['target_dec'] = [-100] * (source_len + 1) + target_tokens + [EOS] + [-100] * total_pad_len
                    data['mask_sub_rel'] = [0] * (source_len + 1) + [1] * (target_len + 1) + [2] * total_pad_len
                else:
                    total_pad_len = fix_len_tgt - source_len - target_len - 1
                    data['length_dec'] = target_len + source_len + 1
                    data['input_dec'] = [SOS] + source_tokens + target_tokens + [EOS] * total_pad_len
                    data['attn_mask_dec'] = [1] * (source_len + target_len + 1) + [0] * total_pad_len
                    data['target_dec'] = source_tokens + target_tokens + [EOS] + [-100] * total_pad_len
                    data['mask_sub_rel'] = [0] * source_len + [1] * (target_len + 1) + [2] * total_pad_len
            else:
                data['length_dec'] = target_len
                data['input_dec'] = [SOS] + target_tokens + [EOS] * target_pad_len
                data['attn_mask_dec'] = [1] * (target_len + 1) + [0] * target_pad_len
                data['target_dec'] = target_tokens + [EOS] + [-100] * target_pad_len
                #data['target_dec'] = target_tokens + [-100] + [-100] * target_pad_len

            dataset.append(data)

        dataset_list.append(dataset)

    return tuple(PCFGDataset(data, vocab) for data in dataset_list)




def PCFG(*args, **kwargs):
    return _setup_pcfg(*args, **kwargs)

class PCFGDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __getitem__(self, i):
        return {'input_enc': np.asarray(self.data[i]['input_enc'], dtype=np.int64),
                     'attn_mask_enc': np.asarray(self.data[i]['attn_mask_enc']),
                     'length_enc': np.asarray(self.data[i]['length_enc']),
                     'input_dec': np.asarray(self.data[i]['input_dec'], dtype=np.int64),
                     'target_dec': np.asarray(self.data[i]['target_dec'], dtype=np.int64),
                     'attn_mask_dec': np.asarray(self.data[i]['attn_mask_dec']),
                     'length_dec': np.asarray(self.data[i]['length_dec']),
                'token_type_ids': np.asarray(self.data[i]['token_type_ids']),
                'mask_sub_rel': np.asarray(self.data[i]['mask_sub_rel'])
                }

    def __iter__(self):
        for i in range(self.__len__()):
            print(i)
            yield self[i]

    def __len__(self):
        return len(self.data)

    def get_pad_token_id(self):
        return -100

    def reverse(self, batch):
        sentences = []
        for indices in batch:
            indices[indices == -100] = 0
            tokens = self.vocab.lookup_tokens(list(indices))
            sentences.append(' '.join(tokens))

        return sentences








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

if __name__ == '__main__':
    train, _, _ = PCFG(fix_len=500, root='/raid/data/pcfg')

    inp_len = train.data[:]['input_len']
    #tgt_len = train[:]['target_len']
    print(inp_len.size())
