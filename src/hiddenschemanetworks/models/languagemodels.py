from abc import ABC, abstractmethod
from typing import Any, Dict

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Gamma

from hiddenschemanetworks.models.blocks_transformers import PseudoSelfAttentionGPT2LMHeadModel
from hiddenschemanetworks.models.helper_functions import clip_grad_norm
from transformers import GPT2LMHeadModel, GPT2Config
from transformers import BertModel, BertConfig


from hiddenschemanetworks.utils.helper import create_instance

class AModel(nn.Module, ABC):
    def __init__(self, **kwargs):
        super().__init__()
        self.reduce = kwargs.get('reduce')
        if 'metrics' in kwargs:
            metrics = create_instance('metrics', kwargs)
            if type(metrics) is not list:
                metrics = [metrics]
            self.metrics = metrics
        else:
            self.metrics = None

    @property
    def device(self):
        return next(self.parameters()).device


class GPT2(AModel):
    """
    GPT2 pretrained language model
    uses DataLoaderPretrained
    """

    def __init__(self, pad_token_id, fix_len, **kwargs):
        super(GPT2, self).__init__(**kwargs)
        self.fix_len = fix_len
        self.ignore_index = pad_token_id


        self.pretrained = kwargs.get('pretrained', True)
        if self.pretrained:
            self.get_logits = GPT2LMHeadModel.from_pretrained('gpt2')
        else:
            config = GPT2Config.from_pretrained('gpt2')
            config.n_embd = kwargs.get('hidden_size', config.n_embd)
            config.n_layer = kwargs.get('num_hidden_layers', config.n_layer)
            config.n_head = kwargs.get('num_attention_heads', config.n_head)
            config.n_inner = kwargs.get('intermediate_size', config.n_inner)
            vocab_size = kwargs.get('vocab_size', None)
            if vocab_size is not None:
                config.vocab_size = vocab_size

            self.get_logits = GPT2LMHeadModel(config=config)

        if self.metrics is not None:
            for m in self.metrics:
                m.ignore_index = self.ignore_index
                m.reduce = self.reduce

        print("---------------")
        print("GPT2")
        print("---------------")


    def forward(self, input_seq, attn_mask):
        """
        input: tuple(data, seq_len), shape: ([B, T], [T])
        Notation. B: batch size; T: seq len (== fix_len)
        """
        logits = self.get_logits(input_ids=input_seq,
                                 attention_mask=attn_mask,
                                 use_cache=False,
                                 output_attentions=False,
                                 output_hidden_states=False,
                                 return_dict=False, )[0]
        return logits


    def loss(self, logits, target_seq, stats):
        """
        returns the loss function of the (discrete) Wasserstein autoencoder
        Notation. B: batch size; T: seq len (== fix_len); L: random walk length
        """

        loss = self._get_reconstruction_error(logits, target_seq)

        stats['NLL-Loss'] = loss
        stats['loss'] = loss

        return stats


    def metric(self, logits, target_seq, seq_len, optim=None):
        """
        returns a dictionary with metrics
        Notation. B: batch size; T: seq len (== fix_len)
        """
        with torch.no_grad():
            batch_size = seq_len.size(0)
            stats = self.new_metric_stats()

            # Number of hits:
            prediction = logits.argmax(dim=-1)
            stats['number_of_hits'] = torch.sum((prediction == target_seq).float()) / float(batch_size)

            # Perplexity:
            cost = self._get_reconstruction_error(logits, target_seq)
            stats['PPL'] = torch.exp(batch_size * cost / torch.sum(seq_len))

            # sequence accuracy:
            pad_mask = (target_seq == self.ignore_index)
            prediction[pad_mask] = 0
            target_seq[pad_mask] = 0
            sequence_distance = torch.sum((prediction != target_seq).float(), dim=1)
            stats['sequence accuracy'] = torch.mean((sequence_distance == 0).float())

            if optim is not None:
                stats['lr'] = torch.tensor(optim['optimizer']['opt'].param_groups[0]['lr'], device=self.device)

            if self.metrics is not None:
                for m in self.metrics:
                    stats[type(m).__name__] = m(logits, target_seq)

            return stats, prediction


    def _get_reconstruction_error(self, y, y_target):
        """
        Notation. B: batch size; T: seq len (== fix_len)
        """

        batch_size = y.size(0)
        y = y.contiguous().view(batch_size * y.size(1), -1)
        y_target = y_target.contiguous().view(-1)

        cost = nn.functional.cross_entropy(y, y_target, ignore_index=self.ignore_index, reduction='sum')
        cost = cost / float(batch_size)

        return cost


    def train_step(self, minibatch: Any, optimizer: Any, step: int, scheduler: Any = None):
        """
        Notation. B: batch size; T: seq len (== fix_len)
        """
        input_seq = minibatch['input_dec']
        target_seq = minibatch['target_dec']
        seq_len = minibatch['length_dec']
        attn_mask = minibatch['attn_mask_dec']

        # Statistics
        stats = self.new_stats()

        self.train()

        # Train loss
        logits = self.forward(input_seq, attn_mask)

        loss_stats = self.loss(logits, target_seq, stats)

        # update lr
        lr = scheduler['lr_scheduler'](step)
        optimizer['optimizer']['opt'].param_groups[0]['lr'] = lr

        optimizer['optimizer']['opt'].zero_grad()
        loss_stats['NLL-Loss'].backward()
        clip_grad_norm(self.parameters(), optimizer['optimizer'])
        optimizer['optimizer']['opt'].step()

        metric_stats, prediction = self.metric(logits, target_seq, seq_len, optim=optimizer)

        return {**loss_stats,
                **metric_stats,
                **{'reconstruction': (prediction, target_seq)}}

    def validate_step(self, minibatch: Any):
        """
        Notation. B: batch size; T: seq len (== fix_len)
        """
        input_seq = minibatch['input_dec']
        target_seq = minibatch['target_dec']
        seq_len = minibatch['length_dec']
        attn_mask = minibatch['attn_mask_dec']

        # Statistics
        stats = self.new_stats()

        # Evaluate model:
        self.eval()

        logits = self.forward(input_seq, attn_mask)

        loss_stats = self.loss(logits, target_seq, stats)


        metric_stats, prediction = self.metric(logits, target_seq, seq_len)

        return {**loss_stats,
                **metric_stats,
                **{'reconstruction': (prediction, target_seq)}}

    def new_stats(self) -> Dict:
        stats = dict()
        stats['loss'] = torch.tensor(0, device=self.device)
        stats['NLL-Loss'] = torch.tensor(0, device=self.device)
        return stats

    def new_metric_stats(self) -> Dict:
        stats = dict()
        stats['PPL'] = torch.tensor(0, device=self.device)
        return stats

class RealSchema(AModel):

    def __init__(self, pad_token_id, fix_len, **kwargs):
        super(RealSchema, self).__init__(**kwargs)
        self.fix_len = fix_len
        self.ignore_index = pad_token_id

        self.n_symbols = kwargs.get('n_symbols')
        self.symbol_dim = self.n_symbols

        self.indices = torch.arange(self.n_symbols).view(1, 1, self.n_symbols).float()

        self.kl_threshold_rw = kwargs.get('kl_threshold_rw', 0.0)
        self.kl_threshold_graph = kwargs.get('kl_threshold_graph', 0.0)
        self.word_dropout = kwargs.get('word_dropout', 0.0)

        # Encoder:
        self.encoder = create_instance('encoder', kwargs, *(self.fix_len,
                                                            self.n_symbols))
        self.rw_length = self.encoder.rw_length

        # Decoder:
        self.decoder = create_instance('decoder', kwargs, *(self.fix_len,
                                                            self.symbol_dim,
                                                            self.rw_length))

        # Symbols:
        self.symbols = nn.Parameter(torch.eye(self.n_symbols), requires_grad=False)

        # Prior:
        self.f_prior = nn.Parameter(torch.randn(self.rw_length-1, self.n_symbols), requires_grad=True)
        torch.nn.init.normal_(self.f_prior, mean=0.0, std=0.01)
        self.f0_prior = nn.Parameter(torch.randn(self.n_symbols), requires_grad=True)
        torch.nn.init.normal_(self.f0_prior, mean=0.5, std=0.01)

        self.softplus = torch.nn.Softplus()

        # Graph generator:
        self.graph_generator = create_instance('graph_generator', kwargs,
                                               *(self.symbols.shape[-1], self.n_symbols))

        self.default_rate = kwargs.get('default_rate', 3.0)
        self.default_shape = np.sqrt(2.0 / (self.graph_generator.n_communities * (self.n_symbols - 1))) \
                             * self.default_rate
        self.edge_prob = kwargs.get('Erdos_edge_prob', 0.5)
        offset = 0 if self.graph_generator.diag_in_adj_matrix else 1
        self.triu_indices = torch.triu_indices(row=self.n_symbols, col=self.n_symbols, offset=offset)


        if self.metrics is not None:
            for m in self.metrics:
                m.ignore_index = self.ignore_index
                m.reduce = self.reduce

        print("---------------")
        print("Pretrained Schemata")
        print("---------------")
        print("Number of Symbols: ", self.n_symbols)
        print("Random walk length: ", self.encoder.rw_length)

    def forward(self, enc_input_seq, dec_input_seq, enc_attn_mask, dec_attn_mask,
                tau_rw=torch.tensor(1.0), tau_graph=torch.tensor(1.0), hard_rw_samples=True, hard_graph_samples=True):
        """
        input: tuple(data, seq_len), shape: ([B, T], [T])
        Notation. B: batch size; T: seq len (== fix_len)
        """

        batch_size = enc_input_seq.shape[0]

        adj_matrix, link_prob, params_graph_model = self.graph_generator(self.symbols, tau_graph, hard=hard_graph_samples)

        _, link_prob_prior = self.sample_prior_graph()

        kl_graph = self.graph_generator.get_kl(link_prob, (link_prob_prior, self.default_shape, self.default_rate),
                                               batch_size)

        # Random walk inference model:

        p_matrix = self._get_prior_prob_trans_matrix_rws(adj_matrix)
        f0_prior = self.f0_prior

        z_post, kl_rws, kl_0, _ = self.encoder((enc_input_seq, enc_attn_mask), adj_matrix, tau_rw,
                                                           (p_matrix, f0_prior),
                                                           hard=hard_rw_samples)  # [B, L, number_symbols]

        # Decoding:
        symbol_seq = torch.matmul(z_post, self.symbols)  # [B, L, symbol_dim]
        logits = self.decoder(dec_input_seq, symbol_seq, dec_attn_mask)

        return logits, z_post, kl_rws, kl_0, kl_graph, adj_matrix

    def train_step(self, minibatch: Any, optimizer: Any, step: int, scheduler: Any = None):
        """
        Notation. B: batch size; T: seq len (== fix_len); L: random walk length
        """
        enc_input_seq = minibatch['input_enc']
        dec_input_seq = minibatch['input_dec']
        target_seq = minibatch['target_dec']
        seq_len = minibatch['length_dec']
        enc_attn_mask = minibatch['attn_mask_enc']
        dec_attn_mask = minibatch['attn_mask_dec']

        # word dropout
        if self.word_dropout > 0:
            word_dropout_mask = torch.empty_like(dec_input_seq).bernoulli_(1 - self.word_dropout)
            dec_attn_mask = word_dropout_mask * dec_attn_mask


        # Statistics
        stats = self.new_stats()

        # schedulers
        beta_rw = torch.tensor(scheduler['beta_scheduler_kl_rws'](step), device=self.device)
        tau_rw = torch.tensor(scheduler['temperature_scheduler_rws'](step), device=self.device)
        beta_graph = torch.tensor(scheduler['beta_scheduler_kl_graph'](step), device=self.device)
        tau_graph = torch.tensor(scheduler['temperature_scheduler_graph'](step), device=self.device)


        # Train loss
        logits, z_post, kl_rws, kl_0, kl_graph, \
        adj_matrix = self.forward(enc_input_seq, dec_input_seq, enc_attn_mask, dec_attn_mask, tau_rw, tau_graph,
                                  hard_graph_samples=False)

        loss_stats = self.loss(logits, target_seq, kl_rws, kl_0, kl_graph, stats, seq_len, beta_rw=beta_rw,
                               beta_graph=beta_graph)

        optimizer['optimizer']['opt'].zero_grad()
        loss_stats['loss'].backward()
        clip_grad_norm(self.parameters(), optimizer['optimizer'])
        optimizer['optimizer']['opt'].step()

        metric_stats, prediction = self.metric(logits, target_seq, kl_rws, kl_0, kl_graph, seq_len, adj_matrix)


        z_post = torch.sum(z_post * self.indices.to(self.device), dim=-1)  # [B, L]

        return {**loss_stats,
                **metric_stats,
                **{'reconstruction': (prediction, dec_input_seq)},
                **{'symbols': z_post}}

    def validate_step(self, minibatch: Any):
        """
        Notation. B: batch size; T: seq len (== fix_len); L: random walk length
        """
        enc_input_seq = minibatch['input_enc']
        dec_input_seq = minibatch['input_dec']
        target_seq = minibatch['target_dec']
        seq_len = minibatch['length_dec']
        enc_attn_mask = minibatch['attn_mask_enc']
        dec_attn_mask = minibatch['attn_mask_dec']

        # Statistics
        stats = self.new_stats()

        # Evaluate model:
        self.eval()
        tau_rw = torch.tensor(0.5, device=self.device)
        tau_graph = torch.tensor(0.5, device=self.device)

        logits, z_post, kl_rws, kl_0, kl_graph, \
        adj_matrix = self.forward(enc_input_seq, dec_input_seq, enc_attn_mask, dec_attn_mask, tau_rw, tau_graph)

        loss_stats = self.loss(logits, target_seq, kl_rws, kl_0, kl_graph, stats, seq_len)

        metric_stats, prediction = self.metric(logits, target_seq, kl_rws, kl_0, kl_graph, seq_len, adj_matrix)

        z_post = torch.sum(z_post * self.indices.to(self.device), dim=-1)  # [B, L]

        return {**loss_stats,
                **metric_stats,
                **{'reconstruction': (prediction, dec_input_seq)},
                **{'symbols': z_post}}

    def loss(self, y, y_target, kl_rws, kl_0, kl_graph,
             stats, seq_len, beta_rw=torch.tensor(1.0), beta_graph=torch.tensor(1.0)):
        """
        returns the loss function of the (discrete) Wasserstein autoencoder
        Notation. B: batch size; T: seq len (== fix_len); L: random walk length
        """

        rec_cost = self._get_reconstruction_error(y, y_target, seq_len)


        loss = rec_cost + max(kl_rws + kl_0, self.kl_threshold_rw) * beta_rw +\
                beta_graph * max(kl_graph, self.kl_threshold_graph)

        stats['KL-RWs'] = kl_rws
        stats['KL-0'] = kl_0
        stats['KL-Graph'] = kl_graph
        stats['weight-KL-RWs'] = beta_rw
        stats['weight-KL-Graph'] = beta_graph
        stats['loss'] = loss
        stats['NLL-Loss'] = rec_cost

        return stats


    def metric(self, y, y_target, kl_rws, kl_0, kl_graph, seq_len, adj_matrix):
        """
        returns a dictionary with metrics
        Notation. B: batch size; T: seq len (== fix_len); L: random walk length
        """
        with torch.no_grad():
            batch_size = seq_len.size(0)
            stats = self.new_metric_stats()

            # Number of hits:
            prediction = y.argmax(dim=-1)
            stats['number_of_hits'] = torch.sum((prediction == y_target).float()) / float(batch_size)

            # sequence accuracy:
            stats['sequence accuracy'] = torch.sum((torch.sum((prediction != y_target).float(), dim=1) == 0).float())

            # Number of edges in graph:
            stats["n_edges_in_graph"] = torch.sum(adj_matrix)

            # Perplexity:
            cost = self._get_reconstruction_error(y, y_target, seq_len)

            kl_graph = batch_size * kl_graph  # since the normalization in "_get_distance_reg"
                                              # is for training only.
            stats['PPL'] = torch.exp(batch_size * (cost + (kl_rws + kl_0) + kl_graph) / torch.sum(seq_len))


            if self.metrics is not None:
                for m in self.metrics:
                    stats[type(m).__name__] = m(y, y_target)

            return stats, prediction


    def _get_reconstruction_error(self, y, y_target, seq_len):
        """
        Notation. B: batch size; T: seq len (== fix_len); L: random walk length
        """
        batch_size = seq_len.size(0)
        y = y.contiguous().view(batch_size * y.size(1), -1)
        y_target = y_target.contiguous().view(-1)
        cost = nn.functional.cross_entropy(y, y_target, ignore_index=self.ignore_index, label_smoothing=0.1)
        return cost


    def sample_prior_graph(self):
        """
        Samples prior graph
        """

        edge_prob = torch.ones([self.n_symbols, self.n_symbols]).to(self.device).fill_(self.edge_prob)

        offset = 0 if self.graph_generator.diag_in_adj_matrix else 1
        p = edge_prob[torch.triu(torch.ones(self.n_symbols, self.n_symbols), diagonal=offset) == 1]
        r = torch.rand(p.shape, device=self.device)
        edges = p > r
        adjacency = torch.zeros(self.n_symbols, self.n_symbols, device=self.device)
        adjacency[self.triu_indices[0], self.triu_indices[1]] = edges.float()
        adjacency = adjacency + torch.transpose(adjacency, 0, 1)
        link_prob = p.view(-1, 1)
        link_prob = torch.cat((link_prob, 1 - link_prob), dim=1)  # [n_symbols*(n_symbols-1)/2, 2]

        return adjacency.float(), link_prob

    def _get_prior_prob_trans_matrix_rws(self, adj_matrix):
        p_matrix = adj_matrix
        f_prior = torch.exp(self.f_prior)
        f_prior = f_prior.unsqueeze(0)
        f_prior_matrix = torch.einsum('bli,blj->blij', f_prior, f_prior)  # [1, L, n_symbols, n_symbols]
        p_matrix = p_matrix.unsqueeze(0).unsqueeze(0) * f_prior_matrix
        p_matrix = self.get_transition_prob_matrix(p_matrix, batched=True).squeeze(0)
        return p_matrix


    def get_transition_prob_matrix(self, matrix, batched=False):
        """
        normalizes symmetric matrix into probability matrix
        """
        pi = torch.full((1, self.n_symbols), 1.0 / self.n_symbols, device=self.device).float()
        if batched:
            # matrix shape: [B, L, n_symbols, n_symbols]
            torch_sum = torch.sum(matrix, dim=-1).view(-1, self.encoder.rw_length - 1, self.n_symbols,
                                                       1)  # [B, n_symbols, 1]
        else:
            # matrix shape: [n_symbols, n_symbols]
            torch_sum = torch.sum(matrix, dim=-1).view(self.n_symbols, 1)  # [n_symbols, 1]
        cond = (torch_sum > 0.0).float()
        norm = torch_sum + (1.0 - cond)
        matrix = cond * (matrix / norm) + (1 - cond) * pi
        return matrix


    def sample_rw_prior(self, batch_size, adj_matrix=None):
        """
        :param adj_matrix: [n_symbols, n_symbols]
        If adj_matrix is NOT None returns rws from fixed input adj_matrix
        else, samples first adj_matrix from prior
        """
        p_matrix = self._get_prior_prob_trans_matrix_rws(adj_matrix).unsqueeze(0)  # [1, n_symbols, n_symbols]
        p_matrix = p_matrix.repeat(batch_size, 1, 1, 1)

        # Distribution over starting point:
        f0 = torch.softmax(self.f0_prior, 0)
        f0 = f0.view(1, self.n_symbols).repeat(batch_size, 1)

        # sample first step:
        cat = torch.distributions.categorical.Categorical(f0)
        z = nn.functional.one_hot(cat.sample(), num_classes=self.n_symbols).float().to(self.device)  # [B, n_symbols]
        walks = torch.unsqueeze(z, 1)  # [B, 1, n_symbols]

        for i in range(1, self.encoder.rw_length):
            # transition prob:
            transition_prob = torch.matmul(z.unsqueeze(1), p_matrix[:, i-1]).squeeze(1)  # [B, n_symbols]
            # (*) sample step
            cat = torch.distributions.categorical.Categorical(transition_prob)
            z = nn.functional.one_hot(cat.sample(), num_classes=self.n_symbols).float()  # [B, n_symbols]
            walks = torch.cat([walks, torch.unsqueeze(z, 1)], dim=1)

        return walks

    def new_stats(self) -> Dict:
        stats = dict()
        stats['loss'] = torch.tensor(0, device=self.device)
        stats['NLL-Loss'] = torch.tensor(0, device=self.device)
        stats['KL-RWs'] = torch.tensor(0, device=self.device)
        stats['KL-0'] = torch.tensor(0, device=self.device)
        stats['KL-Graph'] = torch.tensor(0, device=self.device)
        stats['weight-KL-RWs'] = torch.tensor(0, device=self.device)
        stats['weight-KL-Graph'] = torch.tensor(0, device=self.device)

        return stats

    def new_metric_stats(self) -> Dict:
        stats = dict()
        stats['PPL'] = torch.tensor(0, device=self.device)
        stats['number_of_hits'] = torch.tensor(0, device=self.device)
        stats['symbols'] = torch.tensor(0, device=self.device)
        stats['sequence accuracy'] = torch.tensor(0, device=self.device)
        stats['lr'] = torch.tensor(0, device=self.device)
        return stats

class Translator(AModel):

    def __init__(self, pad_token_id, fix_len, **kwargs):
        super(Translator, self).__init__(**kwargs)
        self.fix_len = fix_len
        self.ignore_index = pad_token_id

        self.word_dropout = kwargs.get('word_dropout', 0.0)

        # Encoder:
        config = BertConfig().from_pretrained('bert-base-uncased')
        encoder_kwargs = kwargs['encoder']['args']
        config.hidden_size = encoder_kwargs.get('hidden_size', config.hidden_size)
        config.num_hidden_layers = encoder_kwargs.get('num_hidden_layers', config.num_hidden_layers)
        config.num_attention_heads = encoder_kwargs.get('num_attention_heads', config.num_attention_heads)
        config.intermediate_size = encoder_kwargs.get('intermediate_size', config.intermediate_size)

        vocab_size = encoder_kwargs.get('vocab_size', None)
        if vocab_size is not None:
            config.vocab_size = vocab_size
        if encoder_kwargs.get('pretrained', True):
            self.get_hidden_states = BertModel.from_pretrained('bert-base-uncased', config=config)
        else:
            self.get_hidden_states = BertModel(config=config)

        # Decoder:
        config = GPT2Config.from_pretrained('gpt2')
        decoder_kwargs = kwargs['decoder']['args']
        config.add_cross_attention = True
        config.n_embd = decoder_kwargs.get('hidden_size', config.n_embd)
        config.n_layer = decoder_kwargs.get('num_hidden_layers', config.n_layer)
        config.n_head = decoder_kwargs.get('num_attention_heads', config.n_head)
        config.n_inner = decoder_kwargs.get('intermediate_size', config.n_inner)
        vocab_size = decoder_kwargs.get('vocab_size', None)
        if vocab_size is not None:
            config.vocab_size = vocab_size
        if decoder_kwargs.get('pretrained', True):
            self.get_logits = GPT2LMHeadModel.from_pretrained('gpt2', config=config)
        else:
            self.get_logits = GPT2LMHeadModel(config=config)

        if self.metrics is not None:
            for m in self.metrics:
                m.ignore_index = self.ignore_index
                m.reduce = self.reduce

        print("---------------")
        print("Translation Transformer")
        print("---------------")

    def forward(self, enc_input_seq, dec_input_seq, enc_attn_mask, dec_attn_mask):
        """
        input: tuple(data, seq_len), shape: ([B, T], [T])
        Notation. B: batch size; T: seq len (== fix_len)
        """

        hidden_states = self.get_hidden_states(input_ids=enc_input_seq,
                                               attention_mask=enc_attn_mask,
                                               token_type_ids=None,
                                               position_ids=None,
                                               head_mask=None,
                                               use_cache=False,
                                               output_attentions=False,
                                               output_hidden_states=False,
                                               return_dict=False, )[0]  # [B, L, D]

        # Decoding:
        logits = self.get_logits(input_ids=dec_input_seq,
                                 encoder_hidden_states=hidden_states,
                                 attention_mask=dec_attn_mask,
                                 token_type_ids=None,
                                 position_ids=None,
                                 head_mask=None,
                                 use_cache=False,
                                 output_attentions=False,
                                 output_hidden_states=False,
                                 return_dict=False, )[0]

        return logits

    def train_step(self, minibatch: Any, optimizer: Any, step: int, scheduler: Any = None):
        """
        Notation. B: batch size; T: seq len (== fix_len); L: random walk length
        """
        enc_input_seq = minibatch['input_enc']
        dec_input_seq = minibatch['input_dec']
        target_seq = minibatch['target_dec']
        enc_attn_mask = minibatch['attn_mask_enc']
        dec_attn_mask = minibatch['attn_mask_dec']

        # word dropout
        if self.word_dropout > 0:
            word_dropout_mask = torch.empty_like(dec_input_seq).bernoulli_(1 - self.word_dropout)
            dec_attn_mask = word_dropout_mask * dec_attn_mask


        # Statistics
        stats = self.new_stats()

        # Train loss
        logits = self.forward(enc_input_seq, dec_input_seq, enc_attn_mask, dec_attn_mask)

        loss_stats = self.loss(logits, target_seq, stats)

        optimizer['optimizer']['opt'].zero_grad()
        # update lr
        lr = scheduler['lr_scheduler'](step)
        optimizer['optimizer']['opt'].param_groups[0]['lr'] = lr
        loss_stats['loss'].backward()
        clip_grad_norm(self.parameters(), optimizer['optimizer'])
        optimizer['optimizer']['opt'].step()

        metric_stats, prediction = self.metric(logits, target_seq, optimizer)

        return {**loss_stats,
                **metric_stats,
                **{'reconstruction': (prediction, dec_input_seq)}}

    def validate_step(self, minibatch: Any):
        """
        Notation. B: batch size; T: seq len (== fix_len); L: random walk length
        """
        enc_input_seq = minibatch['input_enc']
        dec_input_seq = minibatch['input_dec']
        target_seq = minibatch['target_dec']
        enc_attn_mask = minibatch['attn_mask_enc']
        dec_attn_mask = minibatch['attn_mask_dec']

        # Statistics
        stats = self.new_stats()

        # Evaluate model:
        self.eval()

        logits = self.forward(enc_input_seq, dec_input_seq, enc_attn_mask, dec_attn_mask)

        loss_stats = self.loss(logits, target_seq, stats)

        metric_stats, prediction = self.metric(logits, target_seq)

        return {**loss_stats,
                **metric_stats,
                **{'reconstruction': (prediction, dec_input_seq)}}

    def loss(self, y, y_target, stats):
        batch_size = y.size(0)
        y = y.contiguous().view(batch_size * y.size(1), -1)
        y_target = y_target.contiguous().view(-1)
        loss = nn.functional.cross_entropy(y, y_target, ignore_index=self.ignore_index, label_smoothing=0.1)
        stats['loss'] = loss
        stats['NLL-Loss'] = loss
        return stats


    def metric(self, y, y_target, optim=None):
        """
        returns a dictionary with metrics
        Notation. B: batch size; T: seq len (== fix_len); L: random walk length
        """
        with torch.no_grad():
            batch_size = y.size(0)
            stats = self.new_metric_stats()

            # Number of hits:
            prediction = y.argmax(dim=-1)
            stats['number_of_hits'] = torch.sum((prediction == y_target).float()) / float(batch_size)

            # sequence accuracy:
            pad_mask = (y_target == self.ignore_index)
            prediction[pad_mask] = 0
            y_target[pad_mask] = 0
            sequence_distance = torch.sum((prediction != y_target).float(), dim=1)
            stats['sequence accuracy'] = torch.mean((sequence_distance == 0).float())

            if optim is not None:
                stats['lr'] = torch.tensor(optim['optimizer']['opt'].param_groups[0]['lr'], device=self.device)

            if self.metrics is not None:
                for m in self.metrics:
                    stats[type(m).__name__] = m(y, y_target)

            return stats, prediction

    def new_stats(self) -> Dict:
        stats = dict()
        stats['loss'] = torch.tensor(0, device=self.device)
        stats['NLL-Loss'] = torch.tensor(0, device=self.device)
        stats['KL-RWs'] = torch.tensor(0, device=self.device)
        stats['KL-0'] = torch.tensor(0, device=self.device)
        stats['KL-Graph'] = torch.tensor(0, device=self.device)
        stats['weight-KL-RWs'] = torch.tensor(0, device=self.device)
        stats['weight-KL-Graph'] = torch.tensor(0, device=self.device)

        return stats

    def new_metric_stats(self) -> Dict:
        stats = dict()
        stats['PPL'] = torch.tensor(0, device=self.device)
        stats['number_of_hits'] = torch.tensor(0, device=self.device)
        stats['symbols'] = torch.tensor(0, device=self.device)
        stats['lr'] = torch.tensor(0, device=self.device)
        return stats

class TranslatorSchema(RealSchema):
    def __init__(self, pad_token_id, fix_len, **kwargs):
        super(TranslatorSchema, self).__init__(pad_token_id, fix_len, **kwargs)

    def train_step(self, minibatch: Any, optimizer: Any, step: int, scheduler: Any = None):
        """
        Notation. B: batch size; T: seq len (== fix_len); L: random walk length
        """
        enc_input_seq = minibatch['input_enc']
        dec_input_seq = minibatch['input_dec']
        target_seq = minibatch['target_dec']
        seq_len = minibatch['length_dec']
        enc_attn_mask = minibatch['attn_mask_enc']
        dec_attn_mask = minibatch['attn_mask_dec']

        # word dropout
        if self.word_dropout > 0:
            word_dropout_mask = torch.empty_like(dec_input_seq).bernoulli_(1 - self.word_dropout)
            dec_attn_mask = word_dropout_mask * dec_attn_mask

        # Statistics
        stats = self.new_stats()

        # schedulers
        beta_rw = torch.tensor(scheduler['beta_scheduler_kl_rws'](step), device=self.device)
        tau_rw = torch.tensor(scheduler['temperature_scheduler_rws'](step), device=self.device)
        beta_graph = torch.tensor(scheduler['beta_scheduler_kl_graph'](step), device=self.device)
        tau_graph = torch.tensor(scheduler['temperature_scheduler_graph'](step), device=self.device)

        # Train loss
        logits, z_post, kl_rws, kl_0, kl_graph, \
        adj_matrix = self.forward(enc_input_seq, dec_input_seq, enc_attn_mask, dec_attn_mask, tau_rw, tau_graph,
                                  hard_graph_samples=False)

        loss_stats = self.loss(logits, target_seq, kl_rws, kl_0, kl_graph, stats, seq_len, beta_rw=beta_rw,
                               beta_graph=beta_graph)

        optimizer['optimizer']['opt'].zero_grad()
        # update lr
        lr = scheduler['lr_scheduler'](step)
        optimizer['optimizer']['opt'].param_groups[0]['lr'] = lr
        loss_stats['loss'].backward()
        optimizer['optimizer']['opt'].step()

        metric_stats, prediction = self.metric(logits, target_seq, kl_rws, kl_0, kl_graph, seq_len, adj_matrix, optimizer)

        z_post = torch.sum(z_post * self.indices.to(self.device), dim=-1)  # [B, L]


        return {**loss_stats,
                **metric_stats,
                **{'reconstruction': (prediction, dec_input_seq)},
                **{'symbols': z_post}}

    def metric(self, y, y_target, kl_rws, kl_0, kl_graph, seq_len, adj_matrix, optim=None):
        """
        returns a dictionary with metrics
        Notation. B: batch size; T: seq len (== fix_len); L: random walk length
        """
        with torch.no_grad():
            batch_size = seq_len.size(0)
            stats = self.new_metric_stats()

            # Number of hits:
            prediction = y.argmax(dim=-1)
            stats['number_of_hits'] = torch.sum((prediction == y_target).float()) / float(batch_size)

            # sequence accuracy:
            pad_mask = (y_target == self.ignore_index)
            prediction[pad_mask] = 0
            y_target[pad_mask] = 0
            sequence_distance = torch.sum((prediction != y_target).float(), dim=1)
            stats['sequence accuracy'] = torch.mean((sequence_distance == 0).float())

            if optim is not None:
                stats['lr'] = torch.tensor(optim['optimizer']['opt'].param_groups[0]['lr'], device=self.device)

            # Number of edges in graph:
            stats["n_edges_in_graph"] = torch.sum(adj_matrix)

            # Perplexity:
            cost = self._get_reconstruction_error(y, y_target, seq_len)

            kl_graph = batch_size * kl_graph  # since the normalization in "_get_distance_reg"
                                              # is for training only.
            stats['PPL'] = torch.exp(batch_size * (cost + (kl_rws + kl_0) + kl_graph) / torch.sum(seq_len))


            if self.metrics is not None:
                for m in self.metrics:
                    stats[type(m).__name__] = m(y, y_target)

            return stats, prediction


class SyntheticSchema(AModel):
    """
    Simple Schemata
    """

    def __init__(self, data_loader, **kwargs):
        super(SyntheticSchema, self).__init__()

        self.voc_dim = data_loader.number_of_tokens
        self.fix_len = data_loader.sentence_size
        self.n_symbols = data_loader.number_of_schemas
        self.ignore_index = data_loader.vocab.stoi['<pad>']

        self.indices = torch.arange(self.n_symbols).view(1, 1, self.n_symbols).float()

        # True Graph
        self.schemata_networkx = data_loader.schemata_full
        self.schemata_nodes = data_loader.schemata_full.nodes()
        self.adj_matrix = nn.Parameter(torch.tensor(nx.adj_matrix(data_loader.schemata_full,
                                                                  self.schemata_nodes).todense()).float(),
                                       requires_grad=False)

        self.emb_dim = kwargs.get('emb_dim')

        self.kl_threshold_rw = kwargs.get('kl_threshold_rw', 0.0)
        self.kl_threshold_graph = kwargs.get('kl_threshold_graph', 0.0)

        # Encoder:
        self.encoder = create_instance('encoder', kwargs, *(data_loader.vocab,
                                                            self.fix_len,
                                                            None,
                                                            self.n_symbols,
                                                            self.voc_dim,
                                                            self.emb_dim))

        # Decoder:
        self.symbols = nn.Parameter(torch.zeros(self.n_symbols, self.voc_dim), requires_grad=False)
        nn.init.zeros_(self.symbols)

        self.ground_truth_word_prob = nn.Parameter(torch.zeros(self.n_symbols, self.voc_dim), requires_grad=False)
        nn.init.zeros_(self.ground_truth_word_prob)
        schema_words = nx.get_node_attributes(data_loader.schemata_full, "schema_words")
        for schema, words in schema_words.items():
            word_1_index = data_loader.vocab.stoi[words[0].lower()]
            word_2_index = data_loader.vocab.stoi[words[1].lower()]
            self.ground_truth_word_prob[schema, word_1_index] = 0.5
            self.ground_truth_word_prob[schema, word_2_index] = 0.5
            self.symbols[schema, word_1_index] = 0.5
            self.symbols[schema, word_2_index] = 0.5

        self.softplus = torch.nn.Softplus()

        # Graph:
        self.triu_indices = torch.triu_indices(row=self.n_symbols, col=self.n_symbols, offset=1)

        # Graph generator:
        self.graph_generator = create_instance('graph_generator', kwargs,
                                               *(self.symbols.shape[-1], self.n_symbols))

        self.default_rate = kwargs.get('default_rate', 3.0)
        self.default_shape = np.sqrt(2.0 / (self.graph_generator.n_communities * (self.n_symbols - 1))) \
                             * self.default_rate
        self.edge_prob = kwargs.get('Erdos_edge_prob', (self.default_shape / self.default_rate) ** 2)
        if self.edge_prob is None:
            self.edge_prob = (self.default_shape / self.default_rate) ** 2

        if self.metrics is not None:
            for m in self.metrics:
                m.ignore_index = self.ignore_index
                m.reduce = self.reduce


        print("---------------")
        print("Synthetic Schema")
        print("---------------")
        print("Number of Symbols: ", self.n_symbols)
        print("Random walk length: ", self.encoder.rw_length)
        print("Edge probability: ", self.edge_prob)


    def forward(self, input_enc, tau_rw=torch.tensor(1.0), tau_graph=torch.tensor(1.0), z_real=None, hard=True):
        """
        input: tuple(data, seq_len), shape: ([B, T], [T])
        Notation. B: batch size; T: seq len (== fix_len)
        """

        batch_size = input_enc[0].shape[0]

        # Random graph model
        adj_matrix, link_prob, params_graph_model = self.graph_generator(self.symbols,
                                                                           tau_graph,
                                                                           hard=hard)

        weibull_variable = self.sample_gamma_var_prior()
        _, link_prob_prior = self.sample_prior_graph(weibull_variable)

        kl_graph = self.graph_generator.get_kl(link_prob, (link_prob_prior, self.default_shape,
                                                         self.default_rate), batch_size)

        # Random walk inference model:
        p_matrix = self.get_transition_prob_matrix(adj_matrix)
        z_post, kl_rws, kl_0, walk_prob_aux = self.encoder(input_enc, adj_matrix, tau_rw,
                                                           (p_matrix, None),
                                                           z_real, hard=hard)  # [B, L, number_symbols]

        z = z_post


        logits = torch.matmul(z, self.symbols)  # [B, L, V]
        logits = logits.contiguous().view(-1, self.voc_dim)

        return logits, z, kl_rws, kl_0, kl_graph, walk_prob_aux, adj_matrix

    def loss(self, y, y_target, kl_rws, kl_0, kl_graph,
             stats, seq_len, beta_rw=torch.tensor(1.0), beta_graph=torch.tensor(1.0)):
        """
        returns the loss function of the (discrete) Wasserstein autoencoder
        Notation. B: batch size; T: seq len (== fix_len); L: random walk length
        """

        cost = self._get_reconstruction_error(y, y_target, seq_len)

        loss = cost + max(kl_rws + kl_0, self.kl_threshold_rw) * beta_rw + \
               beta_graph * max(kl_graph, self.kl_threshold_graph)

        stats['KL-RWs'] = kl_rws
        stats['KL-0'] = kl_0
        stats['KL-Graph'] = kl_graph
        stats['weight-KL-RWs'] = beta_rw
        stats['weight-KL-Graph'] = beta_graph


        stats['loss'] = loss
        stats['NLL-Loss'] = cost

        return stats

    def metric(self, y, y_target, kl_rws, kl_0, kl_graph, seq_len, walk_prob_aux, z_aux, adj_matrix):
        """
        returns a dictionary with metrics
        Notation. B: batch size; T: seq len (== fix_len); L: random walk length
        """
        with torch.no_grad():
            batch_size = seq_len.size(0)
            stats = self.new_metric_stats()

            # Cross entropy of rws walks:
            if walk_prob_aux is not None:
                walk_prob_aux = walk_prob_aux.view(-1, self.n_symbols)
                z_aux_ = torch.sum(z_aux * self.indices.to(self.device), dim=-1).view(-1).long()  # [B, L]
                cross_entropy_walks = nn.functional.cross_entropy(walk_prob_aux, z_aux_, reduction='sum')
                cross_entropy_walks = cross_entropy_walks / float(batch_size)
            else:
                cross_entropy_walks = torch.tensor(0, device=self.device)

            stats["cross_entropy_walks"] = cross_entropy_walks

            # Number of hits:
            logits_aux = torch.matmul(z_aux, self.ground_truth_word_prob)  # [B, L, V]
            target_seq = torch.nonzero((logits_aux == 0.5), as_tuple=True)[-1]
            target_seq_a, target_seq_b = target_seq.view(batch_size, self.fix_len, -1)[:, :, 0], \
                                         target_seq.view(batch_size, self.fix_len, -1)[:, :, 1]
            prediction = y.argmax(dim=-1).view(batch_size, -1)
            stats['number_of_hits'] = (torch.sum((prediction == target_seq_a).float())
                                       + torch.sum((prediction == target_seq_b).float())) / float(batch_size)

            # Ground-truth cost:
            logits_aux = logits_aux.contiguous().view(-1, self.voc_dim)
            ground_truth_cost = self._get_reconstruction_error(logits_aux, y_target, seq_len)
            stats["ground_truth_cost"] = ground_truth_cost

            # Number of edges in graph:
            stats["n_edges_in_graph"] = torch.sum(adj_matrix)

            # Perplexity:
            cost = self._get_reconstruction_error(y, y_target, seq_len)

            stats['PPL'] = torch.exp(batch_size * (cost + (kl_rws + kl_0) + kl_graph) / torch.sum(seq_len))

            if self.metrics is not None:
                for m in self.metrics:
                    stats[type(m).__name__] = m(y, y_target)

            return stats, prediction, target_seq_a, target_seq_b


    def _get_reconstruction_error(self, y, y_target, seq_len):
        """
        returns the loss function of the (discrete) Wasserstein autoencoder
        Notation. B: batch size; T: seq len (== fix_len); L: random walk length
        """

        batch_size = seq_len.size(0)
        cost = nn.functional.cross_entropy(y, y_target, ignore_index=self.ignore_index, reduction='sum')
        cost = cost / float(batch_size)

        return cost


    def train_step(self, minibatch: Any, optimizer: Any, step: int, scheduler: Any = None):
        """
        Notation. B: batch size; T: seq len (== fix_len); L: random walk length
        """
        input_seq = minibatch.text.long().to(self.device)
        target_seq = minibatch.text.view(-1).long().to(self.device)
        z_real = minibatch.walks.to(self.device)


        B = input_seq.size(0)
        seq_len = torch.ones(B).fill_(self.fix_len)

        z_real = nn.functional.one_hot(z_real.long(), num_classes=self.n_symbols).float()  # [B, L, n_symbols]

        # Statistics
        stats = self.new_stats()

        # schedulers
        beta_rw = torch.tensor(scheduler['beta_scheduler_kl_rws'](step), device=self.device)
        beta_graph = torch.tensor(scheduler['beta_scheduler_kl_graph'](step), device=self.device)
        tau_rw = torch.tensor(scheduler['temperature_scheduler_rws'](step), device=self.device)
        tau_graph = torch.tensor(scheduler['temperature_scheduler_graph'](step), device=self.device)

        # Train loss
        logits, z_post, kl_rws, kl_0, kl_graph, \
        walk_prob_aux, adj_matrix = self.forward((input_seq, seq_len), tau_graph, z_real=z_real, hard=True)


        loss_stats = self.loss(logits, target_seq, kl_rws, kl_0, kl_graph,
                               stats, seq_len, beta_rw=beta_rw, beta_graph=beta_graph)

        optimizer['optimizer']['opt'].zero_grad()
        loss_stats['loss'].backward()
        clip_grad_norm(self.parameters(), optimizer['optimizer'])
        optimizer['optimizer']['opt'].step()

        metric_stats, prediction, \
        target_seq_a, target_seq_b = self.metric(logits, target_seq, kl_rws, kl_0, kl_graph,
                                                 seq_len, walk_prob_aux, z_real, adj_matrix)

        z_post = torch.sum(z_post * self.indices.to(self.device), dim=-1)  # [B, L]
        z_real = torch.sum(z_real * self.indices.to(self.device), dim=-1)

        return {**loss_stats,
                **metric_stats,
                **{'reconstruction': (prediction, target_seq_a, target_seq_b, z_post, z_real)}}

    def validate_step(self, minibatch: Any):
        """
        Notation. B: batch size; T: seq len (== fix_len); L: random walk length
        """
        input_seq = minibatch.text.long().to(self.device)
        target_seq = minibatch.text.view(-1).long().to(self.device)
        z_real = minibatch.walks.to(self.device)

        B = input_seq.size(0)
        seq_len = torch.ones(B).fill_(self.fix_len)

        z_real = nn.functional.one_hot(z_real.long(), num_classes=self.n_symbols).float()  # [B, n_symbols]

        # Statistics
        stats = self.new_stats()

        # Evaluate model:
        tau_rw = torch.tensor(0.5, device=self.device)
        tau_graph = torch.tensor(0.5, device=self.device)

        logits, z_post, kl_rws, kl_0, kl_graph, \
        walk_prob_aux, adj_matrix = self.forward((input_seq, seq_len), tau_graph, z_real=z_real, hard=True)

        loss_stats = self.loss(logits, target_seq, kl_rws, kl_0, kl_graph,
                               stats, seq_len)

        metric_stats, prediction, \
        target_seq_a, target_seq_b = self.metric(logits, target_seq, kl_rws, kl_0, kl_graph,
                                                 seq_len, walk_prob_aux, z_real, adj_matrix)

        z_post = torch.sum(z_post * self.indices.to(self.device), dim=-1)  # [B, L]
        z_real = torch.sum(z_real * self.indices.to(self.device), dim=-1)

        return {**loss_stats,
                **metric_stats,
                **{'reconstruction': (prediction, target_seq_a, target_seq_b, z_post, z_real)},
                **{'symbols': z_post}}

    def sample_prior_graph(self, gamma_var):
        """
        Samples prior graph
        """
        edge_prob = 1.0 - torch.exp(-torch.tensordot(gamma_var, gamma_var, dims=[[1], [1]]))  # [n_symbols, n_symbols]
        offset = 0 if self.graph_generator.diag_in_adj_matrix else 1
        p = edge_prob[torch.triu(torch.ones(self.n_symbols, self.n_symbols), diagonal=offset) == 1]
        r = torch.rand(p.shape, device=self.device)
        edges = p > r
        adjacency = torch.zeros(self.n_symbols, self.n_symbols, device=self.device)
        adjacency[self.triu_indices[0], self.triu_indices[1]] = edges.float()
        adjacency = adjacency + torch.transpose(adjacency, 0, 1)
        link_prob = p.view(-1, 1)
        link_prob = torch.cat((link_prob, 1 - link_prob), dim=1)  # [n_symbols*(n_symbols-1)/2, 2]

        return adjacency.float(), link_prob

    def get_transition_prob_matrix(self, matrix):
        """
        normalizes symmetric matrix into probability matrix
        """
        pi = torch.full((1, self.n_symbols), 1.0 / self.n_symbols, device=self.device).float()
        # matrix shape: [n_symbols, n_symbols]
        torch_sum = torch.sum(matrix, dim=-1).view(self.n_symbols, 1)  # [n_symbols, 1]
        cond = (torch_sum > 0.0).float()
        norm = torch_sum + (1.0 - cond)
        matrix = cond * (matrix / norm) + (1 - cond) * pi
        return matrix

    def sample_rw_prior(self, batch_size, adj_matrix=None):
        """
        :param adj_matrix: [n_symbols, n_symbols]
        If adj_matrix is NOT None returns rws from fixed input adj_matrix
        else, samples first adj_matrix from prior
        """

        # Transition prob matrix:
        if adj_matrix is None:
            w = self.sample_gamma_var_prior()
            adj_matrix = self.sample_prior_graph(w)[0]

        p_matrix = self.get_transition_prob_matrix(adj_matrix).unsqueeze(0)  # [1, n_symbols, n_symbols]
        p_matrix = p_matrix.repeat(batch_size, 1, 1)

        # Distribution over starting point:
        f0 = torch.full((self.n_symbols,), 1.0 / float(self.n_symbols))
        f0 = f0.view(1, self.n_symbols).repeat(batch_size, 1)

        # sample first step:
        cat = torch.distributions.categorical.Categorical(f0)
        z = nn.functional.one_hot(cat.sample(), num_classes=self.n_symbols).float().to(self.device)  # [B, n_symbols]
        walks = torch.unsqueeze(z, 1)  # [B, 1, n_symbols]

        for i in range(1, self.encoder.rw_length):
            # transition prob:
            transition_prob = torch.matmul(z.unsqueeze(1), p_matrix).squeeze(1)  # [B, n_symbols]

            # (*) sample step
            cat = torch.distributions.categorical.Categorical(transition_prob)
            z = nn.functional.one_hot(cat.sample(), num_classes=self.n_symbols).float()  # [B, n_symbols]
            walks = torch.cat([walks, torch.unsqueeze(z, 1)], dim=1)

        return walks

    def new_stats(self) -> Dict:
        stats = dict()
        stats['loss'] = torch.tensor(0, device=self.device)
        stats['NLL-Loss'] = torch.tensor(0, device=self.device)
        stats['KL-RWs'] = torch.tensor(0, device=self.device)
        stats['KL-0'] = torch.tensor(0, device=self.device)
        stats['KL-Graph'] = torch.tensor(0, device=self.device)
        stats['weight-KL-RWs'] = torch.tensor(0, device=self.device)
        stats['weight-KL-Graph'] = torch.tensor(0, device=self.device)

        return stats

    def new_metric_stats(self) -> Dict:
        stats = dict()
        stats['PPL'] = torch.tensor(0, device=self.device)
        stats['cross_entropy_walks'] = torch.tensor(0, device=self.device)
        stats['ground_truth_cost'] = torch.tensor(0, device=self.device)
        return stats

    def sample_gamma_var_prior(self):
        b = torch.ones([self.n_symbols, self.graph_generator.n_communities]).to(self.device).fill_(self.default_rate)
        a = np.sqrt(2.0 / (self.graph_generator.n_communities * (self.n_symbols - 1))) * b
        w = Gamma(a, b).sample()
        return w


class NARRNN(AModel):
    """
    Non autoregressive RNN model
    """

    def __init__(self, data_loader, **kwargs):
        super(NARRNN, self).__init__()

        self.voc_dim = data_loader.number_of_tokens
        self.fix_len = data_loader.sentence_size
        self.n_symbols = data_loader.number_of_schemas
        self.ignore_index = data_loader.vocab.stoi['<pad>']

        self.indices = torch.arange(self.n_symbols).view(1, 1, self.n_symbols).float()

        self.emb_dim = kwargs.get('emb_dim')
        self.rw_length = kwargs.get('rw_length')

        self.encoder = create_instance('encoder', kwargs, *(data_loader.vocab,
                                                            self.fix_len,
                                                            self.n_symbols,
                                                            self.rw_length,
                                                            self.voc_dim,
                                                            self.emb_dim))

        self.decoder_type = None

        self.symbols = nn.Parameter(torch.zeros(self.n_symbols, self.voc_dim), requires_grad=False)
        nn.init.zeros_(self.symbols)

        self.ground_truth_word_prob = nn.Parameter(torch.zeros(self.n_symbols, self.voc_dim), requires_grad=False)
        nn.init.zeros_(self.ground_truth_word_prob)
        schema_words = nx.get_node_attributes(data_loader.schemata_full, "schema_words")
        for schema, words in schema_words.items():
            word_1_index = data_loader.vocab.stoi[words[0].lower()]
            word_2_index = data_loader.vocab.stoi[words[1].lower()]
            self.ground_truth_word_prob[schema, word_1_index] = 0.5
            self.ground_truth_word_prob[schema, word_2_index] = 0.5
            self.symbols[schema, word_1_index] = 0.5
            self.symbols[schema, word_2_index] = 0.5

        self.softplus = torch.nn.Softplus()

        if self.metrics is not None:
            for m in self.metrics:
                m.ignore_index = self.ignore_index
                m.reduce = self.reduce
        print("---------------")
        print("Schemata Baselines: Non AR RNN model")
        print("---------------")

    def forward(self, y, tau, hard=True):
        """
        input: tuple(data, seq_len), shape: ([B, T], [T])
        Notation. B: batch size; T: seq len (== fix_len)
        """

        z_post = self.encoder(y, tau, hard=hard)  # [B, L, number_symbols]

        logits = torch.matmul(z_post, self.symbols)  # [B, L, V]
        logits = logits.contiguous().view(-1, self.voc_dim)

        return logits, z_post

    def loss(self, y, y_target, stats, seq_len):
        """
        returns the loss function
        """

        cost = self._get_reconstruction_error(y, y_target, seq_len)
        stats['loss'] = cost
        stats['NLL-Loss'] = cost

        return stats

    def metric(self, y, y_target, seq_len, z_aux):
        """
        returns a dictionary with metrics
        Notation. B: batch size; T: seq len (== fix_len); L: random walk length
        """
        with torch.no_grad():
            batch_size = seq_len.size(0)
            stats = self.new_metric_stats()

            # Number of hits:
            logits_aux = torch.matmul(z_aux, self.ground_truth_word_prob)  # [B, L, V]
            target_seq = torch.nonzero((logits_aux == 0.5), as_tuple=True)[-1]
            target_seq_a, target_seq_b = target_seq.view(batch_size, self.fix_len, -1)[:, :, 0], \
                                         target_seq.view(batch_size, self.fix_len, -1)[:, :, 1]
            prediction = y.argmax(dim=-1).view(batch_size, -1)
            stats['number_of_hits'] = (torch.sum((prediction == target_seq_a).float())
                                       + torch.sum((prediction == target_seq_b).float())) / float(batch_size)

            # Ground-truth cost:
            logits_aux = logits_aux.contiguous().view(-1, self.voc_dim)
            ground_truth_cost = self._get_reconstruction_error(logits_aux, y_target, seq_len)
            stats["ground_truth_cost"] = ground_truth_cost

            if self.metrics is not None:
                for m in self.metrics:
                    stats[type(m).__name__] = m(y, y_target)

            return stats, prediction, target_seq_a, target_seq_b


    def _get_reconstruction_error(self, y, y_target, seq_len):
        """
        returns the loss function of the (discrete) Wasserstein autoencoder
        Notation. B: batch size; T: seq len (== fix_len); L: random walk length
        """
        batch_size = seq_len.size(0)
        cost = nn.functional.cross_entropy(y, y_target, ignore_index=self.ignore_index, reduction='sum')
        cost = cost / float(batch_size)
        return cost


    def train_step(self, minibatch: Any, optimizer: Any, step: int, scheduler: Any = None):
        """
        Notation. B: batch size; T: seq len (== fix_len); L: random walk length
        """
        input_seq = minibatch.text.long().to(self.device)
        target_seq = minibatch.text.view(-1).long().to(self.device)
        z_real = minibatch.walks.to(self.device)

        B = input_seq.size(0)
        seq_len = torch.ones(B).fill_(self.fix_len)

        z_real = nn.functional.one_hot(z_real.long(), num_classes=self.n_symbols).float()  # [B, L, n_symbols]

        # Statistics
        stats = self.new_stats()

        tau_rw = torch.tensor(scheduler['temperature_scheduler_rws'](step), device=self.device)

        # Initialize hidden state for rnn models
        self.initialize_hidden_state(B, self.device)

        # with torch.autograd.detect_anomaly():

        # Train loss
        logits, z_post = self.forward((input_seq, seq_len), tau_rw, hard=True)


        loss_stats = self.loss(logits, target_seq, stats, seq_len)

        optimizer['optimizer']['opt'].zero_grad()
        loss_stats['loss'].backward()
        clip_grad_norm(self.parameters(), optimizer['optimizer'])
        optimizer['optimizer']['opt'].step()

        metric_stats, prediction, \
        target_seq_a, target_seq_b = self.metric(logits, target_seq, seq_len, z_real)

        self.detach_history()

        z_post = torch.sum(z_post * self.indices.to(self.device), dim=-1)  # [B, L]
        z_real = torch.sum(z_real * self.indices.to(self.device), dim=-1)

        return {**loss_stats,
                **metric_stats,
                **{'reconstruction': (prediction, target_seq_a, target_seq_b, z_post, z_real)}}


    def validate_step(self, minibatch: Any):
        """
        Notation. B: batch size; T: seq len (== fix_len); L: random walk length
        """
        input_seq = minibatch.text.long().to(self.device)
        target_seq = minibatch.text.view(-1).long().to(self.device)
        z_real = minibatch.walks.to(self.device)

        B = input_seq.size(0)
        seq_len = torch.ones(B).fill_(self.fix_len)

        z_real = nn.functional.one_hot(z_real.long(), num_classes=self.n_symbols).float()  # [B, n_symbols]

        # Statistics
        stats = self.new_stats()

        # Initialize hidden state for rnn models
        self.initialize_hidden_state(B, self.device)

        # Evaluate model:
        tau_rw = torch.tensor(0.5, device=self.device)

        logits, z_post = self.forward((input_seq, seq_len), tau_rw, hard=True)

        loss_stats = self.loss(logits, target_seq, stats, seq_len)

        metric_stats, prediction, \
        target_seq_a, target_seq_b = self.metric(logits, target_seq, seq_len, z_real)

        z_post = torch.sum(z_post * self.indices.to(self.device), dim=-1)  # [B, L]
        z_real = torch.sum(z_real * self.indices.to(self.device), dim=-1)

        return {**loss_stats,
                **metric_stats,
                **{'reconstruction': (prediction, target_seq_a, target_seq_b, z_post, z_real)},
                **{'symbols': z_post}}



    def new_stats(self) -> Dict:
        stats = dict()
        stats['loss'] = torch.tensor(0, device=self.device)
        stats['NLL-Loss'] = torch.tensor(0, device=self.device)
        return stats

    def new_metric_stats(self) -> Dict:
        stats = dict()
        return stats

    def initialize_hidden_state(self, batch_size, device, enc=True, dec=True):
        if enc and self.encoder.is_recurrent:
            self.encoder.initialize_hidden_state(batch_size, device)

    def detach_history(self, enc=True, dec=True):
        if self.encoder.is_recurrent and enc:
            self.encoder.reset_history()
