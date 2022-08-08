from abc import ABC, abstractmethod
import networkx as nx
import os

import torch
from nltk.tokenize import TweetTokenizer
from torch.utils.data import DataLoader
import pickle
import numpy as np
from scipy import sparse as sp
import matplotlib.pyplot as plt


from hiddenschemanetworks.data.datasets import PennTreebank, YelpReview, YahooAnswers,\
    SyntheticSchemataDataset, PCFG

sampler = torch.utils.data.RandomSampler

DistributedSampler = torch.utils.data.distributed.DistributedSampler

tokenizer = TweetTokenizer(preserve_case=False).tokenize


class ADataLoader(ABC):
    def __init__(self, device, rank: int = 0, world_size: int = -1, **kwargs):
        self.device = device
        self.batch_size = kwargs.pop('batch_size')
        self.path_to_vectors = kwargs.pop('path_to_vectors', None)
        self.emb_dim = kwargs.pop('emb_dim', None)
        self.voc_size = kwargs.pop('voc_size', None)
        self.min_freq = kwargs.pop('min_freq', 1)
        self._fix_length = kwargs.pop('fix_len', None)
        self.min_len = kwargs.pop('min_len', None)
        self.max_len = kwargs.pop('max_len', None)
        self.lower = kwargs.pop('lower', False)
        self.punctuation = kwargs.pop('punctuation', True)
        self.dataset_kwargs = kwargs
        self.world_size = world_size
        self.rank = rank

    @property
    @abstractmethod
    def train(self): ...

    @property
    @abstractmethod
    def validate(self): ...

    @property
    @abstractmethod
    def test(self): ...

    @property
    def n_train_batches(self):
        return len(self.train)

    @property
    def n_validate_batches(self):
        return len(self.validate)

    @property
    def n_test_batches(self):
        return len(self.test)

    @property
    def train_set_size(self):
        return len(self.train.dataset)

    @property
    def validation_set_size(self):
        return len(self.validate.dataset)

    @property
    def test_set_size(self):
        return len(self.test.dataset)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class DataLoaderSynthetic(ADataLoader):
    def __init__(self, device: str = "cpu", rank: int = 0, world_size=-1, schemata_name="barabasi", **kwargs):
        path_to_data = kwargs.pop('path_to_data')
        super().__init__(device, rank, world_size, **kwargs)
        self.dataset_kwargs = kwargs
        self.schemata_name = schemata_name
        train_dataset = SyntheticSchemataDataset(path_to_data, "train", schemata_name)
        test_dataset = SyntheticSchemataDataset(path_to_data, "test", schemata_name)
        valid_dataset = SyntheticSchemataDataset(path_to_data, "val", schemata_name)

        self.walk_lenght = train_dataset[0].walks.shape[0]
        self.sentence_size = train_dataset[0].text.shape[0]

        voc = pickle.load(open(os.path.join(path_to_data, "voca.pkl"), "rb"))
        self.read_graphs(path_to_data)

        train_sampler = None
        valid_sampler = None
        test_sampler = None
        if self.world_size != -1:
            train_sampler = DistributedSampler(train_dataset, self.world_size, self.rank)
            test_sampler = DistributedSampler(test_dataset, self.world_size, self.rank)
            valid_sampler = DistributedSampler(valid_dataset, self.world_size, self.rank)

        self._train_iter = DataLoader(train_dataset, drop_last=True, sampler=train_sampler, shuffle=train_sampler is None, **kwargs)
        self._valid_iter = DataLoader(valid_dataset, drop_last=True, sampler=valid_sampler, shuffle=valid_sampler is None, **kwargs)
        self._test_iter = DataLoader(test_dataset, drop_last=True, sampler=test_sampler, shuffle=test_sampler is None, **kwargs)
        self.train_vocab = voc
        self.number_of_tokens = len(voc.itos)

        self.number_of_documents_train = len(train_dataset)
        self.number_of_documents_test = len(test_dataset)
        self.number_of_documents_val = len(valid_dataset)

        self.number_of_documents = self.number_of_documents_train + self.number_of_documents_test + self.number_of_documents_val


    @property
    def train(self):
        return self._train_iter

    @property
    def test(self):
        return self._test_iter

    @property
    def validate(self):
        return self._valid_iter

    @property
    def vocab(self):
        return self.train_vocab

    @property
    def schemata(self):
        return self.schemata

    @property
    def fix_len(self):
        return self._fix_length

    def read_graphs(self,data_dir):
        schema_words = pickle.load(open(os.path.join(data_dir, "schema_words.pkl"), "rb"))
        adjacency_full = pickle.load(open(os.path.join(data_dir, "adjacency_schemata.pkl"), "rb"))
        nodes_full = pickle.load(open(os.path.join(data_dir, "nodes_schemata.pkl"), "rb"))

        training_nodes = pickle.load(
            open(os.path.join(data_dir, "{0}.train_schemata_graph_nodes.gp".format(self.schemata_name)), "rb"))
        test_nodes = pickle.load(
            open(os.path.join(data_dir, "{0}.test_schemata_graph_nodes.gp".format(self.schemata_name)), "rb"))
        validation_nodes = pickle.load(
            open(os.path.join(data_dir, "{0}.valid_schemata_graph_nodes.gp".format(self.schemata_name)), "rb"))

        self.schemata_full = nx.from_numpy_array(adjacency_full)
        print(self.schemata_full)
        self.schemata_full = nx.relabel_nodes(self.schemata_full, dict(zip(range(adjacency_full.shape[0]), nodes_full)))
        nx.set_node_attributes(self.schemata_full, schema_words, "schema_words")

        self.number_of_schemas = len(self.schemata_full.nodes())

        self.train_graph = nx.subgraph(self.schemata_full, training_nodes)
        self.test_graph = nx.subgraph(self.schemata_full, test_nodes)
        self.val_graph = nx.subgraph(self.schemata_full, validation_nodes)

        self.schemata_nodes = self.schemata_full.nodes()
        adj_matrix = nx.adj_matrix(self.schemata_full, self.schemata_nodes).todense()
        adj_matrix = np.asarray(adj_matrix, dtype=float)
        self.schemata_adjacency = torch.Tensor(adj_matrix)
        self.schemata_adjacency_sparse = sparse_mx_to_torch_sparse_tensor(sp.csc_matrix(self.schemata_adjacency))
        self.empty_features = torch.ones(self.number_of_schemas)

class DataLoaderText(ADataLoader):

    def __init__(self, device, rank: int = 0, world_size=-1, **kwargs):

        super().__init__(device, rank, world_size, **kwargs)

        path_to_data = kwargs.pop('path_to_data')
        self._fix_len = kwargs.pop('fix_len', 256)


        train_dataset, test_dataset, valid_dataset = self.get_datasets(path_to_data)

        train_sampler = None
        valid_sampler = None
        test_sampler = None
        if self.world_size != -1:
            train_sampler = DistributedSampler(train_dataset, self.world_size, self.rank)
            valid_sampler = DistributedSampler(valid_dataset, self.world_size, self.rank)
            test_sampler = DistributedSampler(test_dataset, self.world_size, self.rank)

        self._train_iter = DataLoader(train_dataset, drop_last=True, sampler=train_sampler,
                                      shuffle=train_sampler is None, **kwargs)
        self._valid_iter = DataLoader(valid_dataset, drop_last=True, sampler=valid_sampler,
                                      shuffle=valid_sampler is None, **kwargs)
        self._test_iter = DataLoader(test_dataset, drop_last=True, sampler=test_sampler,
                                     shuffle=test_sampler is None, **kwargs)

        self._pad_token_id = train_dataset.get_pad_token_id()

        if not isinstance(self, DataLoaderPCFG):
            self._tokenizer = train_dataset.tokenizer_dec

    @property
    def train(self):
        return self._train_iter

    @property
    def test(self):
        return self._test_iter

    @property
    def validate(self):
        return self._valid_iter

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def pad_token_id(self):
        return self._pad_token_id

    @property
    def fix_len(self):
        return self._fix_length

    @property
    def vocab(self): # for compatibility with TextTrainer
        return None

class DataLoaderPennTreebank(DataLoaderText):
    """
    Data loader for PTB with pretrained tokenizers and models from huggingface
    """
    def __init__(self, device, rank: int = 0, world_size=-1, **kwargs):

        super().__init__(device, rank, world_size, **kwargs)

    def get_datasets(self, path_to_data):
        return PennTreebank(root=path_to_data,
                              fix_len=self._fix_len)

class DataLoaderYahooAnswers(DataLoaderText):
    """
    Data loader for YahooAnswers with pretrained tokenizers and models from huggingface
    """

    def __init__(self, device, rank: int = 0, world_size=-1, **kwargs):
        super().__init__(device, rank, world_size, **kwargs)

    def get_datasets(self, path_to_data):
        return YahooAnswers(root=path_to_data,
                              fix_len=self._fix_len)

class DataLoaderYelpReview(DataLoaderText):
    """
    Data loader for YelpReview with pretrained tokenizers and models from huggingface
    """

    def __init__(self, device, rank: int = 0, world_size=-1, **kwargs):
        super().__init__(device, rank, world_size, **kwargs)

    def get_datasets(self, path_to_data):
        return YelpReview(root=path_to_data,
                          fix_len=self._fix_len)


class DataLoaderPCFG(DataLoaderText):
    """
    Data loader for PTB with pretrained tokenizers and models from huggingface
    """
    def __init__(self, device, rank: int = 0, world_size=-1, **kwargs):
        self.atomic_style = kwargs.pop('atomic_style', False)
        self.ae_style = kwargs.pop('ae_style', False)
        self.train_on_object_only = kwargs.pop('train_on_object_only', True)
        super().__init__(device, rank, world_size, **kwargs)

    def get_datasets(self, path_to_data):
        return PCFG(root=path_to_data, atomic_style=self.atomic_style, ae_style=self.ae_style,
                    train_on_object_only=self.train_on_object_only)

    @property
    def vocab(self):
        return self._train_iter.dataset.vocab

    @property
    def tokenizer(self):
        return None

if __name__ == '__main__':
    loader = DataLoaderPCFG(torch.device('cuda:0'), path_to_data='/raid/data/pcfg', batch_size=100,
                            atomic_style=True)

    inp_lens = []
    for mb in loader.train:
        inp_lens.append(mb['length_dec'])

    inp_lens = torch.cat(inp_lens, dim=0)
    bins, counts = torch.unique(inp_lens, sorted=True, return_counts=True)

    plt.scatter(bins.numpy(), counts.numpy())
    plt.show()
    print(bins[-10:])
    print(counts[-10:])



