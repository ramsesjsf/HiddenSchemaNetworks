from typing import Any, Dict, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Gamma

from hiddenschemanetworks.models.helper_functions import clip_grad_norm
from hiddenschemanetworks.models.languagemodels import AModel
from hiddenschemanetworks.utils.helper import create_instance

class RealSchemata(AModel):
    """
    Schemata Model for real data sets
    """

    def __init__(self, vocab, fix_len, **kwargs):
        super(RealSchemata, self).__init__(**kwargs)
        self.voc_dim = vocab.vectors.size(0)
        self.fix_len = fix_len
        self.ignore_index = vocab.stoi['<pad>']

        self.emb_dim = kwargs.get('emb_dim')

        self.loss_type = kwargs.get('loss_type')
        self.symbol_dim = kwargs.get('symbol_dim')
        self.n_symbols = kwargs.get('n_symbols')

        self.indices = torch.arange(self.n_symbols).view(1, 1, self.n_symbols).float()


        self.decoder_type = kwargs.get('decoder_type', "matrix")
        self.train_rw = kwargs.get('train_rw', True)
        self.train_prior = kwargs.get('train_prior', False)
        self.max_entropy_prior = kwargs.get('max_entropy_prior', False)
        self.diversity_reg = kwargs.get('diversity_reg', False)
        self.lambda_diversity = kwargs.get('lambda_diversity', 1.0)
        self.l1_reg = kwargs.get('l1_reg', False)
        self.lambda_l1 = kwargs.get('lambda_l1', 1.0)
        self.word_dropout = kwargs.get('word_dropout', 0.0)
        self.kl_threshold_rw = kwargs.get('kl_threshold_rw', 0.0)
        self.kl_threshold_graph = kwargs.get('kl_threshold_graph', 0.0)

        self.aggregated_post = True if self.loss_type == "VAE-MI" else False

        # Encoder:
        self.encoder = create_instance('encoder', kwargs, *(vocab,
                                                            self.fix_len,
                                                            None,
                                                            self.n_symbols,
                                                            self.aggregated_post,
                                                            self.max_entropy_prior,
                                                            self.voc_dim,
                                                            self.emb_dim))

        # Decoder:

        self.symbols = nn.Parameter(torch.randn(self.n_symbols, self.symbol_dim), requires_grad=True)
        torch.nn.init.normal_(self.symbols, mean=0.0, std=0.01)
        self.decoder = create_instance('decoder', kwargs, *(vocab,
                                                            self.fix_len,
                                                            self.voc_dim,
                                                            self.symbol_dim,
                                                            self.emb_dim))
        self.decoder.embedding = self.encoder.embedding

        # Prior:
        if self.train_prior:
            self.f_prior = nn.Parameter(torch.randn(self.n_symbols), requires_grad=True)
            torch.nn.init.normal_(self.f_prior, mean=0.0, std=0.01)
            self.f0_prior = nn.Parameter(torch.randn(self.n_symbols), requires_grad=True)
            torch.nn.init.normal_(self.f0_prior, mean=0.0, std=0.01)
        else:
            self.f_prior = None
            self.f0_prior = None

        self.softplus = torch.nn.Softplus()

        # Graph:
        self.hard_graph_samples = kwargs.get('hard_graph_samples', True)
        # Graph generator:
        self.train_graph = kwargs.get('train_graph')
        if self.train_graph:
            self.graph_generator = create_instance('graph_generator', kwargs,
                                                   *(self.symbols.shape[-1], self.n_symbols))

            self.default_rate = kwargs.get('default_rate', 3.0)
            self.default_shape = np.sqrt(2.0 / (self.graph_generator.n_communities * (self.n_symbols - 1))) \
                                 * self.default_rate
            self.edge_prob = kwargs.get('Erdos_edge_prob', 0.5)
        else:
            self.adj_matrix = nn.Parameter(torch.ones((self.n_symbols, self.n_symbols), dtype=torch.float),
                                           requires_grad=False)
        offset = 0 if self.graph_generator.diag_in_adj_matrix else 1
        self.triu_indices = torch.triu_indices(row=self.n_symbols, col=self.n_symbols, offset=offset)

        if self.metrics is not None:
            for m in self.metrics:
                m.ignore_index = self.ignore_index
                m.reduce = self.reduce

        # Eigenvalues of graph:
        if self.max_entropy_prior:
            out_eig = torch.eig(self.adj_matrix, eigenvectors=True)
            self.eigenval = nn.Parameter(out_eig[0][:, 0], requires_grad=False)
            self.eigenvec = nn.Parameter(out_eig[1], requires_grad=False)

        print("---------------")
        print("Simple Schemata")
        print("---------------")
        print("Loss type: ", self.loss_type)
        print("Number of Symbols: ", self.n_symbols)
        print("Random walk length: ", self.encoder.rw_length)


    def forward(self, input_seq, seq_len, tau_rw=torch.tensor(1.0), tau_graph=torch.tensor(1.0),
                hard_rw_samples=True, hard_graph_samples=True):
        """
        input: tuple(data, seq_len), shape: ([B, T], [T])
        Notation. B: batch size; T: seq len (== fix_len)
        """

        batch_size = input_seq.shape[0]

        # Random graph model
        if self.train_graph:
            adj_matrix, link_prob, weibull_variable, params_graph_model = self.graph_generator(self.symbols,
                                                                                               tau_graph,
                                                                                               batch_size,
                                                                                               hard=hard_graph_samples)

            _, link_prob_prior = self.sample_prior_graph(weibull_variable)

            kl_graph, kl_weibull_gamma = self.graph_generator.get_kl(link_prob,
                                                                     (link_prob_prior,
                                                                      self.default_shape,
                                                                      self.default_rate),
                                                                     params_graph_model,
                                                                     batch_size)
        else:
            adj_matrix = self.adj_matrix
            kl_graph, kl_weibull_gamma = torch.tensor(0.0, device=self.device), \
                                         torch.tensor(0.0, device=self.device)

        # Random walk inference model:
        p_matrix = self._get_prior_prob_trans_matrix_rws(adj_matrix)
        z_post, kl_rws, kl_0, _ = self.encoder((input_seq, seq_len), adj_matrix, tau_rw,
                                                           (p_matrix, self.f0_prior),
                                                           hard=hard_rw_samples)  # [B, L, number_symbols]

        # Decoding:
        symbol_seq = torch.matmul(z_post, self.symbols)  # [B, L, symbol_dim]
        logits = self.decoder((input_seq, seq_len), symbol_seq)

        return logits, z_post, kl_rws, kl_0, kl_graph, kl_weibull_gamma, adj_matrix


    def loss(self, y, y_target, kl_rws, kl_0, kl_graph, kl_weibull_gamma,
             stats, seq_len, beta_rw=torch.tensor(1.0), beta_graph=torch.tensor(1.0), beta_gamma=torch.tensor(1.0),
             mask_sub_rel=None):
        """
        returns the loss function of the (discrete) Wasserstein autoencoder
        Notation. B: batch size; T: seq len (== fix_len); L: random walk length
        """
        if mask_sub_rel is not None:
            y_target[mask_sub_rel == 0] = self.ignore_index


        rec_cost = self._get_reconstruction_error(y, y_target, seq_len)

        if self.loss_type == "AE":
            loss = rec_cost

        elif self.loss_type in ("VAE", "VAE-MI"):
            loss = rec_cost + max(kl_rws + kl_0, self.kl_threshold_rw) * beta_rw +\
                   beta_graph * max(kl_graph, self.kl_threshold_graph) + beta_gamma * kl_weibull_gamma

            stats['KL-RWs'] = kl_rws
            stats['KL-0'] = kl_0
            stats['KL-Graph'] = kl_graph
            stats['weight-KL-RWs'] = beta_rw
            stats['weight-KL-Graph'] = beta_graph
        else:
            print("loss_type not specified. Please select: {AE, VAE, VAE-MI} ")
            raise Exception

        stats['loss'] = loss
        stats['NLL-Loss'] = rec_cost

        return stats


    def metric(self, y, y_target, kl_rws, kl_0, kl_graph, kl_weibull_gamma, seq_len, adj_matrix, mask_sub_rel, optim=None):
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

            # Number of edges in graph:
            stats["n_edges_in_graph"] = torch.sum(adj_matrix)

            # Perplexity:
            cost = self._get_reconstruction_error(y, y_target, seq_len)
            if self.loss_type == "AE":
                stats['PPL'] = torch.exp(batch_size * cost / torch.sum(seq_len))

            elif self.loss_type in ("VAE", "VAE-MI"):
                kl_graph = batch_size * kl_graph  # since the normalization in "_get_distance_reg"
                                                  # is for training only.
                kl_weibull_gamma = batch_size * kl_weibull_gamma
                stats['PPL'] = torch.exp(batch_size * (cost + (kl_rws + kl_0) + kl_graph +
                                                       kl_weibull_gamma) / torch.sum(seq_len))
            else:
                print("loss_type not specified. Please select: {AE, VAE, VAE-MI} ")
                raise Exception

            # sequence accuracy:
            pad_mask = (mask_sub_rel != 1)
            prediction[pad_mask] = 0
            y_target[pad_mask] = 0
            sequence_distance = torch.sum((prediction != y_target).float(), dim=1)
            stats['sequence accuracy'] = torch.mean((sequence_distance == 0).float())

            if optim is not None:
                stats['lr'] = torch.tensor(optim.param_groups[0]['lr'], device=self.device)

            if self.metrics is not None:
                for m in self.metrics:
                    stats[type(m).__name__] = m(y, y_target)

            return stats, prediction


    def _get_reconstruction_error(self, y, y_target, seq_len):
        """
        returns the loss function of the (discrete) Wasserstein autoencoder
        Notation. B: batch size; T: seq len (== fix_len); L: random walk length
        """

        batch_size = seq_len.size(0)
        y = y.contiguous().view(batch_size * y.size(1), -1)
        y_target = y_target.contiguous().view(-1)

        cost = nn.functional.cross_entropy(y, y_target, ignore_index=self.ignore_index, reduction='sum')
        cost = cost / float(batch_size)

        return cost


    def train_step(self, minibatch: Any, optimizer: Any, step: int, scheduler: Any = None):
        """
        Notation. B: batch size; T: seq len (== fix_len); L: random walk length
        """
        input_seq = minibatch['input']
        target_seq = minibatch['target']
        seq_len = minibatch['length'].cpu()

        # Statistics
        stats = self.new_stats()

        # schedulers
        beta_rw = torch.tensor(scheduler['beta_scheduler_kl_rws'](step), device=self.device)
        tau_rw = torch.tensor(scheduler['temperature_scheduler_rws'](step), device=self.device)
        beta_graph = torch.tensor(scheduler['beta_scheduler_kl_graph'](step), device=self.device)
        tau_graph = torch.tensor(scheduler['temperature_scheduler_graph'](step), device=self.device)


        # Train loss
        logits, z_post, kl_rws, kl_0, kl_graph, kl_weibull_gamma, \
        adj_matrix = self.forward(input_seq, seq_len, tau_rw, tau_graph, hard_graph_samples=self.hard_graph_samples)

        loss_stats = self.loss(logits, target_seq, kl_rws, kl_0, kl_graph, kl_weibull_gamma,
                               stats, seq_len, beta_rw=beta_rw, beta_graph=beta_graph)

        optimizer['optimizer']['opt'].zero_grad()
        loss_stats['loss'].backward()
        clip_grad_norm(self.parameters(), optimizer['optimizer'])
        optimizer['optimizer']['opt'].step()

        metric_stats, prediction = self.metric(logits, target_seq, kl_rws, kl_0, kl_graph, kl_weibull_gamma,
                                                 seq_len, adj_matrix)


        z_post = torch.sum(z_post * self.indices.to(self.device), dim=-1)  # [B, L]

        return {**loss_stats,
                **metric_stats,
                **{'reconstruction': (prediction, target_seq)},
                **{'symbols': z_post}}

    def validate_step(self, minibatch: Any):
        """
        Notation. B: batch size; T: seq len (== fix_len); L: random walk length
        """
        input_seq = minibatch['input']
        target_seq = minibatch['target']
        seq_len = minibatch['length'].cpu()

        # Statistics
        stats = self.new_stats()

        # Evaluate model:
        self.eval()
        tau_rw = torch.tensor(0.5, device=self.device)
        tau_graph = torch.tensor(0.5, device=self.device)

        logits, z_post, kl_rws, kl_0, kl_graph, kl_weibull_gamma, \
        adj_matrix = self.forward(input_seq, seq_len, tau_rw, tau_graph)

        loss_stats = self.loss(logits, target_seq, kl_rws, kl_0, kl_graph, kl_weibull_gamma,
                               stats, seq_len)

        metric_stats, prediction = self.metric(logits, target_seq, kl_rws, kl_0, kl_graph, kl_weibull_gamma,
                                                 seq_len, adj_matrix)

        z_post = torch.sum(z_post * self.indices.to(self.device), dim=-1)  # [B, L]

        return {**loss_stats,
                **metric_stats,
                **{'reconstruction': (prediction, target_seq)},
                **{'symbols': z_post}}

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


    def sample_gamma_var_prior(self):
        b = torch.ones([self.n_symbols, self.graph_generator.n_communities]).to(self.device).fill_(self.default_rate)
        a = np.sqrt(2.0 / (self.graph_generator.n_communities * (self.n_symbols - 1))) * b
        w = Gamma(a, b).sample()
        return w


    def _get_prior_prob_trans_matrix_rws(self, adj_matrix, batch_size=None):
        if self.max_entropy_prior and not self.train_graph:
            index_max_eigval = torch.sort(self.eigenval)[1][-1]
            max_eigenvec = self.eigenvec[:, index_max_eigval]
            p_matrix = adj_matrix * torch.einsum('i,j->ij', max_eigenvec, max_eigenvec)
        else:
            p_matrix = adj_matrix

        if self.train_prior:
            if self.mlp_prior_layers is not None:
                noise = torch.randn(batch_size, self.mlp_prior_layers[0], device=self.device)
                f = self.get_prior(noise)
                f = torch.exp(f).view(-1, self.rw_length, self.n_symbols)
                f_prior_matrix = torch.einsum('bli,blj->blij', f[:, 1:], f[:, 1:])  # [B, L-1, n_symbols, n_symbols]
                p_matrix = p_matrix.unsqueeze(0).unsqueeze(0) * f_prior_matrix
                p_matrix = self.get_transition_prob_matrix(p_matrix, batched=True)
                return p_matrix, f[:, 0]
            else:
                f_prior = torch.exp(self.f_prior)
                if f_prior.dim() == 2:
                    f_prior = f_prior.unsqueeze(0)
                    f_prior_matrix = torch.einsum('bli,blj->blij', f_prior, f_prior)  # [1, L, n_symbols, n_symbols]
                    p_matrix = p_matrix.unsqueeze(0).unsqueeze(0) * f_prior_matrix
                    p_matrix = self.get_transition_prob_matrix(p_matrix, batched=True).squeeze(0)
                else:
                    f_prior_matrix = torch.einsum('i,j->ij', f_prior, f_prior)  # [n_symbols, n_symbols]
                    p_matrix = p_matrix * f_prior_matrix
                    p_matrix = self.get_transition_prob_matrix(p_matrix)
        else:
            p_matrix = self.get_transition_prob_matrix(p_matrix)

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

        # Transition prob matrix:
        if adj_matrix is None:
            w = self.sample_gamma_var_prior()
            adj_matrix = self.sample_prior_graph(w)[0]

        p_matrix = self._get_prior_prob_trans_matrix_rws(adj_matrix).unsqueeze(0)  # [1, n_symbols, n_symbols]

        if self.train_prior:
            p_matrix = p_matrix.repeat(batch_size, 1, 1, 1)
        else:
            p_matrix = p_matrix.repeat(batch_size, 1, 1)

        # Distribution over starting point:
        f0 = torch.softmax(self.f0_prior, 0) if self.train_prior else \
            torch.full((self.n_symbols,), 1.0 / float(self.n_symbols))
        f0 = f0.view(1, self.n_symbols).repeat(batch_size, 1)

        # sample first step:
        cat = torch.distributions.categorical.Categorical(f0)
        z = nn.functional.one_hot(cat.sample(), num_classes=self.n_symbols).float().to(self.device)  # [B, n_symbols]
        walks = torch.unsqueeze(z, 1)  # [B, 1, n_symbols]

        for i in range(1, self.encoder.rw_length):
            # transition prob:
            if self.train_prior:
                transition_prob = torch.matmul(z.unsqueeze(1), p_matrix[:, i-1]).squeeze(1)  # [B, n_symbols]
            else:
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
        stats['KL-weibull_gamma'] = torch.tensor(0, device=self.device)
        stats['weight-KL-RWs'] = torch.tensor(0, device=self.device)
        stats['weight-KL-Graph'] = torch.tensor(0, device=self.device)
        stats['weight-KL-Gamma'] = torch.tensor(0, device=self.device)

        return stats

    def new_metric_stats(self) -> Dict:
        stats = dict()
        stats['PPL'] = torch.tensor(0, device=self.device)
        stats['number_of_hits'] = torch.tensor(0, device=self.device)
        stats['symbols'] = torch.tensor(0, device=self.device)
        stats['lr'] = torch.tensor(0, device=self.device)
        return stats


class RealSchemataPretrained(RealSchemata):

    def __init__(self, pad_token_id, fix_len, **kwargs):
        super(RealSchemata, self).__init__(**kwargs)
        self.fix_len = fix_len
        self.ignore_index = pad_token_id

        self.loss_type = kwargs.get('loss_type')
        self.onehot_symbols = kwargs.get('onehot_symbols', False)
        self.n_symbols = kwargs.get('n_symbols')
        if self.onehot_symbols:
            self.symbol_dim = self.n_symbols
        else:
            self.symbol_dim = kwargs.get('symbol_dim')

        self.indices = torch.arange(self.n_symbols).view(1, 1, self.n_symbols).float()

        self.train_rw = kwargs.get('train_rw', True)
        self.train_prior = kwargs.get('train_prior', False)
        self.mlp_prior_layers = kwargs.get("mlp_prior_layers", None)
        self.max_entropy_prior = kwargs.get('max_entropy_prior', False)
        self.diversity_reg = kwargs.get('diversity_reg', False)
        self.lambda_diversity = kwargs.get('lambda_diversity', 1.0)
        self.l1_reg = kwargs.get('l1_reg', False)
        self.lambda_l1 = kwargs.get('lambda_l1', 1.0)
        self.kl_threshold_rw = kwargs.get('kl_threshold_rw', 0.0)
        self.kl_threshold_graph = kwargs.get('kl_threshold_graph', 0.0)
        self.word_dropout = kwargs.get('word_dropout', 0.0)

        self.aggregated_post = True if self.loss_type == "VAE-MI" else False

        # Encoder:
        self.encoder = create_instance('encoder', kwargs, *(self.fix_len,
                                                            self.n_symbols,
                                                            self.aggregated_post,
                                                            self.max_entropy_prior))
        self.rw_length = self.encoder.rw_length

        # Decoder:
        self.decoder = create_instance('decoder', kwargs, *(self.fix_len,
                                                            self.symbol_dim,
                                                            self.rw_length))

        # Symbols:
        if self.onehot_symbols:
            self.symbols = nn.Parameter(torch.eye(self.n_symbols), requires_grad=False)
        else:
            self.symbols = nn.Parameter(torch.randn(self.n_symbols, self.symbol_dim), requires_grad=True)
            torch.nn.init.normal_(self.symbols, mean=0.0, std=0.01)

        # Prior:
        if self.train_prior:
            if self.mlp_prior_layers is not None:
                dims = self.mlp_prior_layers
                layers = nn.ModuleList([])
                for i in range(len(dims)-1):
                    layers.append(nn.Linear(dims[i], dims[i+1]))
                    layers.append(nn.ReLU())
                layers.append(nn.Linear(dims[-1], self.n_symbols * self.rw_length))
                self.get_prior = nn.Sequential(*layers)
            else:
                self.f_prior = nn.Parameter(torch.randn(self.rw_length-1, self.n_symbols), requires_grad=True)
                torch.nn.init.normal_(self.f_prior, mean=0.0, std=0.01)
                self.f0_prior = nn.Parameter(torch.randn(self.n_symbols), requires_grad=True)
                torch.nn.init.normal_(self.f0_prior, mean=0.5, std=0.01)
        else:
            self.f_prior = None
            self.f0_prior = None

        self.softplus = torch.nn.Softplus()

        # Graph generator:
        self.hard_graph_samples = kwargs.get('hard_graph_samples', True)
        self.train_graph = kwargs.get('train_graph')
        if self.train_graph:
            self.graph_generator = create_instance('graph_generator', kwargs,
                                                   *(self.symbols.shape[-1], self.n_symbols))

            self.default_rate = kwargs.get('default_rate', 3.0)
            self.default_shape = np.sqrt(2.0 / (self.graph_generator.n_communities * (self.n_symbols - 1))) \
                                 * self.default_rate
            self.edge_prob = kwargs.get('Erdos_edge_prob', 0.5)
            offset = 0 if self.graph_generator.diag_in_adj_matrix else 1
            self.triu_indices = torch.triu_indices(row=self.n_symbols, col=self.n_symbols, offset=offset)

        else:
            self.adj_matrix = nn.Parameter(torch.ones((self.n_symbols, self.n_symbols), dtype=torch.float),
                                           requires_grad=False)

        if self.metrics is not None:
            for m in self.metrics:
                m.ignore_index = self.ignore_index
                m.reduce = self.reduce

        # Eigenvalues of graph:
        if self.max_entropy_prior:
            out_eig = torch.eig(self.adj_matrix, eigenvectors=True)
            self.eigenval = nn.Parameter(out_eig[0][:, 0], requires_grad=False)
            self.eigenvec = nn.Parameter(out_eig[1], requires_grad=False)

        print("---------------")
        print("Pretrained Schemata")
        print("---------------")
        print("Loss type: ", self.loss_type)
        print("Number of Symbols: ", self.n_symbols)
        print("Random walk length: ", self.encoder.rw_length)

    def freeze_decoder(self):
        """
        Freeze all decoder parameters but those in
        (1) loading_info['missing_keys']
        (2) transformer.wte, transformer.wpe
        """
        for p in self.decoder.get_logits.transformer.named_parameters():
            if p[0] in self.decoder.loading_info['missing_keys']:
                p[1].requires_grad = True
            else:
                p[1].requires_grad = False
        self.decoder.get_logits.transformer.wte.weight.requires_grad = True
        self.decoder.get_logits.transformer.wpe.weight.requires_grad = True

    def free_decoder(self):
        """
        Frees all decoder parameters but those in
        """
        for p in self.decoder.get_logits.transformer.named_parameters():
            p[1].requires_grad = True

    def forward(self, enc_input_seq, dec_input_seq, enc_attn_mask, dec_attn_mask,
                tau_rw=torch.tensor(1.0), tau_graph=torch.tensor(1.0), hard_rw_samples=True, hard_graph_samples=True):
        """
        input: tuple(data, seq_len), shape: ([B, T], [T])
        Notation. B: batch size; T: seq len (== fix_len)
        """

        batch_size = enc_input_seq.shape[0]

        # Random graph model
        adj_matrix, link_prob, params_graph_model = self.graph_generator(self.symbols,
                                                                               tau_graph,
                                                                               hard=hard_graph_samples)

        _, link_prob_prior = self.sample_prior_graph()

        kl_graph = self.graph_generator.get_kl(link_prob, (link_prob_prior, self.default_shape, self.default_rate),
                                               batch_size)


        # Random walk inference model:
        if self.train_prior and self.mlp_prior_layers is not None:
            p_matrix, f0_prior = self._get_prior_prob_trans_matrix_rws(adj_matrix, batch_size)
        else:
            p_matrix = self._get_prior_prob_trans_matrix_rws(adj_matrix)
            f0_prior = self.f0_prior

        z_post, kl_rws, kl_0, _ = self.encoder((enc_input_seq, enc_attn_mask), adj_matrix, tau_rw,
                                                           (p_matrix, f0_prior),
                                                           hard=hard_rw_samples)  # [B, L, number_symbols]

        # Decoding:
        symbol_seq = torch.matmul(z_post, self.symbols)  # [B, L, symbol_dim]
        logits = self.decoder(dec_input_seq, symbol_seq, dec_attn_mask)

        return logits, z_post, kl_rws, kl_0, kl_graph, 0, adj_matrix

    def train_step(self, minibatch: Any, optimizer: Any, step: int, scheduler: Any = None):
        """
        Notation. B: batch size; T: seq len (== fix_len); L: random walk length
        """
        enc_input_seq = minibatch['input_0']
        dec_input_seq = minibatch['input_1']
        target_seq = minibatch['target_1']
        seq_len = minibatch['length_1'].cpu()
        enc_attn_mask = minibatch['attn_mask_0']
        dec_attn_mask = minibatch['attn_mask_1']

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
        logits, z_post, kl_rws, kl_0, kl_graph, kl_weibull_gamma, \
        adj_matrix = self.forward(enc_input_seq, dec_input_seq, enc_attn_mask, dec_attn_mask, tau_rw, tau_graph,
                                  hard_graph_samples=self.hard_graph_samples)

        loss_stats = self.loss(logits, target_seq, kl_rws, kl_0, kl_graph, kl_weibull_gamma,
                               stats, seq_len, beta_rw=beta_rw, beta_graph=beta_graph)

        optimizer['optimizer']['opt'].zero_grad()
        loss_stats['loss'].backward()
        clip_grad_norm(self.parameters(), optimizer['optimizer'])
        optimizer['optimizer']['opt'].step()

        metric_stats, prediction = self.metric(logits, target_seq, kl_rws, kl_0, kl_graph, kl_weibull_gamma,
                                                 seq_len, adj_matrix)


        z_post = torch.sum(z_post * self.indices.to(self.device), dim=-1)  # [B, L]

        return {**loss_stats,
                **metric_stats,
                **{'reconstruction': (prediction, dec_input_seq)},
                **{'symbols': z_post}}


    def validate_step(self, minibatch: Any):
        """
        Notation. B: batch size; T: seq len (== fix_len); L: random walk length
        """
        enc_input_seq = minibatch['input_0']
        dec_input_seq = minibatch['input_1']
        target_seq = minibatch['target_1']
        seq_len = minibatch['length_1'].cpu()
        enc_attn_mask = minibatch['attn_mask_0']
        dec_attn_mask = minibatch['attn_mask_1']

        # Statistics
        stats = self.new_stats()

        # Evaluate model:
        self.eval()
        tau_rw = torch.tensor(0.5, device=self.device)
        tau_graph = torch.tensor(0.5, device=self.device)

        logits, z_post, kl_rws, kl_0, kl_graph, kl_weibull_gamma, \
        adj_matrix = self.forward(enc_input_seq, dec_input_seq, enc_attn_mask, dec_attn_mask, tau_rw, tau_graph)

        loss_stats = self.loss(logits, target_seq, kl_rws, kl_0, kl_graph, kl_weibull_gamma,
                               stats, seq_len)

        metric_stats, prediction = self.metric(logits, target_seq, kl_rws, kl_0, kl_graph, kl_weibull_gamma,
                                                 seq_len, adj_matrix)

        z_post = torch.sum(z_post * self.indices.to(self.device), dim=-1)  # [B, L]

        return {**loss_stats,
                **metric_stats,
                **{'reconstruction': (prediction, dec_input_seq)},
                **{'symbols': z_post}}


class RealSchemataAtomic2(RealSchemataPretrained):

    def __init__(self, pad_token_id, fix_len, **kwargs):
        super(RealSchemataAtomic2, self).__init__(pad_token_id,
                                                  fix_len,
                                                  **kwargs)
        self.train_on_object_only = kwargs.get('train_on_object_only', False)

    def train_step(self, minibatch: Any, optimizer: Any, step: int, scheduler: Any = None):
        """
        Notation. B: batch size; T: seq len (== fix_len); L: random walk length
        """

        enc_input_seq = minibatch['input_enc']
        dec_input_seq = minibatch['input_dec']
        target_seq = minibatch['target_dec']
        seq_len = minibatch['length_dec'].cpu()
        attn_mask_enc = minibatch['attn_mask_enc']
        attn_mask_dec = minibatch['attn_mask_dec']
        token_type_ids = minibatch['token_type_ids']
        mask_sub_rel = minibatch['mask_sub_rel']

        # word dropout
        if self.word_dropout > 0:
            word_dropout_mask = torch.empty_like(dec_input_seq).bernoulli_(1 - self.word_dropout)
            attn_mask_dec = word_dropout_mask * attn_mask_dec

        # Statistics
        stats = self.new_stats()

        # schedulers
        beta_rw = torch.tensor(scheduler['beta_scheduler_kl_rws'](step), device=self.device)
        tau_rw = torch.tensor(scheduler['temperature_scheduler_rws'](step), device=self.device)
        beta_graph = torch.tensor(scheduler['beta_scheduler_kl_graph'](step), device=self.device)
        tau_graph = torch.tensor(scheduler['temperature_scheduler_graph'](step), device=self.device)

        # Train loss
        logits, z_post, kl_rws, kl_0, kl_graph, kl_weibull_gamma, \
        adj_matrix = self.forward(enc_input_seq,
                                  dec_input_seq,
                                  (attn_mask_enc, token_type_ids),
                                  attn_mask_dec,
                                  tau_rw,
                                  tau_graph,
                                  hard_graph_samples=self.hard_graph_samples)

        ### train on object sequence only
        target_loss = target_seq if self.train_on_object_only else target_seq

        loss_stats = self.loss(logits,
                               target_loss,
                               kl_rws, kl_0, kl_graph, kl_weibull_gamma,
                               stats,
                               seq_len,
                               beta_rw=beta_rw, beta_graph=beta_graph)

        # update lr
        if lr_scheduler := scheduler.get('lr_scheduler', None) is not None:
            lr = lr_scheduler(step)
            optimizer['optimizer']['opt'].param_groups[0]['lr'] = lr

        optimizer['optimizer']['opt'].zero_grad()
        loss_stats['loss'].backward()
        clip_grad_norm(self.parameters(), optimizer['optimizer'])
        optimizer['optimizer']['opt'].step()

        metric_stats, prediction = self.metric(logits, target_seq, kl_rws, kl_0, kl_graph, kl_weibull_gamma,
                                               seq_len, adj_matrix, mask_sub_rel, optimizer['optimizer']['opt'])

        z_post = torch.sum(z_post * self.indices.to(self.device), dim=-1)  # [B, L]

        return {**loss_stats,
                **metric_stats,
                **{'reconstruction': (prediction, target_seq)},
                **{'symbols': z_post}}


    def validate_step(self, minibatch: Any):
        """
        Notation. B: batch size; T: seq len (== fix_len); L: random walk length
        """
        enc_input_seq = minibatch['input_enc']
        dec_input_seq = minibatch['input_dec']
        target_seq = minibatch['target_dec']
        seq_len = minibatch['length_dec'].cpu()
        attn_mask_enc = minibatch['attn_mask_enc']
        attn_mask_dec = minibatch['attn_mask_dec']
        token_type_ids = minibatch['token_type_ids']
        mask_sub_rel = minibatch['mask_sub_rel']

        # Statistics
        stats = self.new_stats()

        # Evaluate model:
        self.eval()
        tau_rw = torch.tensor(0.5, device=self.device)
        tau_graph = torch.tensor(0.5, device=self.device)

        logits, z_post, kl_rws, kl_0, kl_graph, kl_weibull_gamma, \
        adj_matrix = self.forward(enc_input_seq,
                                  dec_input_seq,
                                  (attn_mask_enc, token_type_ids),
                                  attn_mask_dec,
                                  tau_rw,
                                  tau_graph)

        loss_stats = self.loss(logits,
                               target_seq,
                               kl_rws, kl_0, kl_graph, kl_weibull_gamma,
                               stats,
                               seq_len, mask_sub_rel=mask_sub_rel)

        metric_stats, prediction = self.metric(logits, target_seq, kl_rws, kl_0, kl_graph, kl_weibull_gamma,
                                                 seq_len, adj_matrix, mask_sub_rel)

        z_post = torch.sum(z_post * self.indices.to(self.device), dim=-1)  # [B, L]


        return {**loss_stats,
                **metric_stats,
                **{'reconstruction': (prediction, target_seq)},
                **{'symbols': z_post}}