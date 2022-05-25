import torch as torch
from torch import nn as nn
from hiddenschemanetworks.utils.helper import gumbel_softmax, get_class_nonlinearity, greedy_sample_categorical
import math

class Block(nn.Module):
    def __init__(self,
                 vocab=None,
                 fix_len=None,
                 latent_dim=None,
                 recurrent=False,
                 input_dim=0,
                 get_embeddings=True,
                 voc_dim=None,
                 emb_dim=None,
                 use_huggingface_models=False,
                 **kwargs):

        super(Block, self).__init__()
        self.input_dim = input_dim
        self.fix_len = fix_len
        self.latent_dim = latent_dim
        self.recurrent = recurrent

        if not use_huggingface_models:
            self.voc_dim, self.emb_dim = vocab.vectors.size() if vocab.vectors is not None else (voc_dim, emb_dim)
            self.SOS = vocab.stoi['<sos>']
            self.EOS = vocab.stoi['<eos>']
            self.PAD = vocab.stoi['<pad>']
            self.UNK = vocab.unk_index
            self.custom_init = kwargs.get('custom_init')

            if self.voc_dim is None or self.emb_dim is None:
                print("voc_dim or emb_dim not define")
                raise Exception

            if get_embeddings:
                self.embedding = nn.Embedding(self.voc_dim, self.emb_dim)
                if vocab.vectors is not None:
                    emb_matrix = vocab.vectors.to(self.device)
                    self.embedding.weight.data.copy_(emb_matrix)
                    self.embedding.weight.requires_grad = kwargs.get('train_word_embeddings', False)
                else:
                    self.embedding.weight.data.normal_(mean=0.0, std=0.01)
                    self.embedding.weight.requires_grad = True
    @property
    def is_recurrent(self):
        return self.recurrent

    @property
    def device(self):
        return next(self.parameters()).device

    def param_init(self):
        """
        Parameters initialization.
        """
        if self.custom_init:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
                    if module.bias.data is not None:
                        nn.init.zeros_(module.bias)
                if isinstance(module, nn.LSTM):
                    # input-hidden weights:
                    nn.init.xavier_normal_(module.weight_ih_l0[:module.hidden_size])
                    nn.init.xavier_normal_(module.weight_ih_l0[module.hidden_size:2 * module.hidden_size])
                    nn.init.xavier_normal_(module.weight_ih_l0[2 * module.hidden_size:3 * module.hidden_size])
                    nn.init.xavier_normal_(module.weight_ih_l0[-module.hidden_size:])
                    # hidden-hidden weights:
                    nn.init.orthogonal_(module.weight_hh_l0[:module.hidden_size])
                    nn.init.orthogonal_(module.weight_hh_l0[module.hidden_size:2 * module.hidden_size])
                    nn.init.orthogonal_(module.weight_hh_l0[2 * module.hidden_size:3 * module.hidden_size])
                    nn.init.orthogonal_(module.weight_hh_l0[-module.hidden_size:])
                    if module.bias:
                        nn.init.zeros_(module.bias_ih_l0)
                        nn.init.zeros_(module.bias_hh_l0)
                        nn.init.ones_(module.bias_hh_l0[module.hidden_size:2 * module.hidden_size])
                    if module.bidirectional:
                        # input-hidden weights:
                        nn.init.xavier_normal_(module.weight_ih_l0_reverse[:module.hidden_size])
                        nn.init.xavier_normal_(module.weight_ih_l0_reverse[module.hidden_size:2 * module.hidden_size])
                        nn.init.xavier_normal_(module.weight_ih_l0_reverse[2 * module.hidden_size:3 * module.hidden_size])
                        nn.init.xavier_normal_(module.weight_ih_l0_reverse[-module.hidden_size:])
                        # hidden-hidden weights:
                        nn.init.orthogonal_(module.weight_hh_l0_reverse[:module.hidden_size])
                        nn.init.orthogonal_(module.weight_hh_l0_reverse[module.hidden_size:2 * module.hidden_size])
                        nn.init.orthogonal_(module.weight_hh_l0_reverse[2 * module.hidden_size:3 * module.hidden_size])
                        nn.init.orthogonal_(module.weight_hh_l0_reverse[-module.hidden_size:])
                        if module.bias:
                            nn.init.zeros_(module.bias_ih_l0_reverse)
                            nn.init.zeros_(module.bias_hh_l0_reverse)
                            nn.init.ones_(module.bias_hh_l0_reverse[module.hidden_size:2 * module.hidden_size])

class PositionalEncoding(nn.Module):

    def __init__(self, dim, max_len=1000, dropout=0.1, device=None):
        """
        Modified from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.nn.Parameter(torch.zeros(max_len, dim), requires_grad=False)
        position = torch.nn.Parameter(torch.arange(0, max_len, dtype=torch.float), requires_grad=False).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, dim]
        self.register_buffer('pe', pe)

    def forward(self, z):
        """
        z: [B, L, D]
        """
        rw_len = z.size(1)
        z = z + self.pe[:, :rw_len]
        return self.dropout(z)

class EncodeOntoRW(Block):
    def __init__(self, vocab, fix_len, latent_dim, n_symbols,
                 recurrent, voc_dim=None, emb_dim=None, use_huggingface_models=False, **kwargs):
        super(EncodeOntoRW, self).__init__(vocab, fix_len, latent_dim,
                                           recurrent=recurrent, voc_dim=voc_dim, emb_dim=emb_dim,
                                           use_huggingface_models=use_huggingface_models, **kwargs)


        self.rw_length = kwargs.get("rw_length")
        self.const_lower_value_f_matrix = kwargs.get("const_lower_value_f_matrix", 0.0)
        self.softplus = torch.nn.Softplus()
        self.n_symbols = n_symbols


    def forward(self, input, adj_matrix, tau, prior_params, z_aux=None, hard=True):
        """
        input: input data (x): [B, T]
               adj_matrix: tuple( [n_symbols, n_symbols]; [n_symbols]; [n_symbols, n_symbols])
               temperature Gumbel distribution (tau): int
               prior_params: tuple([n_symbols, n_symbols], [n_symbols])

        Returns: one-hot sequence over symbols (z): [B, L, n_symbols]
        Notation:
        B: batch size; T: seq len (== fix_len); L: random walk length; Z: latent_dim
        """

        p_matrix, f0_prior = prior_params

        f0, f = self.get_functions_over_nodes(input)  # [B, n_symbols], [B, L-1, n_symbols]



        # (Unnormalized) Posterior transition prob matrix
        f_matrix = torch.einsum('bli,blj->blij', f, f)  # Batched outer product: [B, L-1, n_symbols, n_symbols]

        f_matrix = f_matrix + self.const_lower_value_f_matrix * torch.ones_like(f_matrix)

        if adj_matrix.dim() == 3:
            q_matrix = adj_matrix.unsqueeze(1) * f_matrix  # [B, L-1, n_symbols, n_symbols]
        else:
            q_matrix = adj_matrix.view(1, 1, self.n_symbols,
                                       self.n_symbols) * f_matrix  # [B, L-1, n_symbols, n_symbols]

        walks, walk_prob, q_matrix = self.sample_walks(q_matrix, f0, tau, z_aux, hard=hard)

        walk_prob, walk_prob_aux = walk_prob

        kl_rws, kl_0 = self.get_kl_rws(walk_prob, q_matrix, p_matrix, f0_prior)

        return walks, kl_rws, kl_0, walk_prob_aux


    def get_functions_over_nodes(self, input):
        pass


    def get_kl_rws(self, walk_prob, q_matrix, p_matrix, f0_prior):
        """
        walk_prob: [B, L, n_symbols] or [L, n_symbols]
        q_matrix:  [B, L-1, n_symbols, n_symbols]
        p_matrix: [n_symbols, n_symbols]
        f0_prior: [n_symbols]
        """

        log_K = torch.log(torch.tensor(self.n_symbols).to(self.device).float())

        # posterior log transition prob:
        cond = (q_matrix > 0.0).float()
        epsilon = torch.ones(q_matrix.shape).fill_(1e-8).to(self.device)
        log_q_matrix = torch.log(torch.max(q_matrix + (1 - cond), epsilon))

        # prior log transition prob:
        if p_matrix.dim() == 4:
            p_matrix = torch.mean(p_matrix, 0)     # [L-1, n_symbols, n_symbols]
            f0_prior = torch.mean(f0_prior, 0)     # [n_symbols]

        cond = (p_matrix > 0.0).float()
        epsilon = torch.ones(p_matrix.shape).fill_(1e-8).to(self.device)
        log_p_matrix = torch.log(torch.max(p_matrix + (1 - cond), epsilon))

        # kl starting point:
        rho_0 = walk_prob[0]  # [n_symbols]
        cond = (rho_0 > 0.0).float()
        epsilon = torch.ones(rho_0.shape).fill_(1e-8).to(self.device)
        log_rho_0 = torch.log(torch.max(rho_0 + (1 - cond), epsilon))
        if f0_prior is not None:
            log_pi_0 = torch.log(torch.softmax(f0_prior, 0))
            kl_0 = torch.sum(rho_0 * (log_rho_0 - log_pi_0))
        else:
            kl_0 = torch.sum(rho_0 * log_rho_0) + log_K

        # kl full walks:
        weight = walk_prob[:-1].view(-1, self.n_symbols, 1) * q_matrix  # [L-1, n_symbols, n_symbols]
        if log_p_matrix.dim() == 3:
            kl_rws = torch.sum(weight * (log_q_matrix - log_p_matrix))
        else:
            kl_rws = torch.sum(weight * (log_q_matrix - log_p_matrix.view(1, self.n_symbols, self.n_symbols)))

        return kl_rws, kl_0

    def sample_walks(self, q_matrix, f0, tau, z_aux=None, hard=True, greedy=False):
        """
        q_matrix: [B, L, n_symbols, n_symbols]
        f0: [B, n_symbols]
        z: [B, n_symbols]
        tau: int
        """

        q_matrix = self.get_transition_prob_matrix(q_matrix, True)
        averaged_q_matrix = torch.mean(q_matrix, 0)  # [L-1, n_symbols, n_symbols]

        # sample first step:
        z = gumbel_softmax(f0, tau, self.device, hard=hard) if not greedy \
            else greedy_sample_categorical(f0)      # [B, n_symbols]

        walks = torch.unsqueeze(z, 1)  # [B, 1, n_symbols]

        # get prob over (aggregated) walks:
        f0_averaged = torch.mean(f0, 0)  # [n_symbols]
        walk_prob = torch.unsqueeze(f0_averaged, 0)  # [1, n_symbols]
        walk_prob_aux = None

        # random walks:
        for i in range(1, self.rw_length):

            # transition prob:
            transition_prob = torch.matmul(z.unsqueeze(1), q_matrix[:, i - 1, :]).squeeze(1)  # [B, n_symbols]

            # (*) sample step
            z = gumbel_softmax(transition_prob, tau, self.device, hard=hard) if not greedy \
                else greedy_sample_categorical(transition_prob)

            walks = torch.cat([walks, torch.unsqueeze(z, 1)], dim=1)

            # MARGINAL (aggr) prob over walks (for regularization)
            f0_averaged = torch.matmul(f0_averaged, averaged_q_matrix[i - 1, :])  # [n_symbols]
            walk_prob = torch.cat([walk_prob, torch.unsqueeze(f0_averaged, 0)], dim=0)  # [i+1, n_symbols]

            if z_aux is not None:
                if i == 1:
                    walk_prob_aux = torch.unsqueeze(f0, 1)
                walk_prob_aux = self._get_auxiliar_walk_prob(i, z_aux, walk_prob_aux, q_matrix[:, i - 1, :])

        walk_prob = (walk_prob, walk_prob_aux)

        return walks, walk_prob, averaged_q_matrix  # [B, L, n_symbols], [L, n_symbols], [L-1, n_symbols, n_symbols]


    def get_transition_prob_matrix(self, matrix, batched=False):
        """
        normalizes symmetric matrix into probability matrix
        """
        pi = torch.full((1, self.n_symbols), 1.0 / self.n_symbols, device=self.device).float()
        if batched:
            # matrix shape: [B, L, n_symbols, n_symbols]
            torch_sum = torch.sum(matrix, dim=-1).view(-1, self.rw_length-1, self.n_symbols, 1)  # [B, n_symbols, 1]
        else:
            # matrix shape: [n_symbols, n_symbols]
            torch_sum = torch.sum(matrix, dim=-1).view(self.n_symbols, 1)  # [n_symbols, 1]
        cond = (torch_sum > 0.0).float()
        norm = torch_sum + (1.0 - cond)
        matrix = cond * (matrix / norm) + (1 - cond) * pi
        return matrix


    def _get_auxiliar_walk_prob(self, i, z, walk_prob, q_matrix):
        """
        returns the walk_probabilities for a predefined walks
        z: [B, L, n_symbols]
        """
        transition_prob = torch.matmul(z[:, i - 1, :].unsqueeze(1), q_matrix).squeeze(1)  # [B, n_symbols]
        walk_prob = torch.cat([walk_prob, torch.unsqueeze(transition_prob, 1)], dim=1)
        return walk_prob

    @property
    def output_size(self):
        return self.__output_size

class GraphGenerator(nn.Module):
    """
    Graph generator
    """
    def __init__(self, emb_input_dim, n_symbols, **kwargs):
        super(GraphGenerator, self).__init__()

        dims = kwargs.get('symbols2hidden_layers')
        nonlinearity = kwargs.get('nonlinearity')
        normalization = kwargs.get('normalization')

        self.n_communities = kwargs.get('n_communities')
        self.diag_in_adj_matrix = kwargs.get('diag_in_adj_matrix', False)
        self.symbol_pair2link_function = kwargs.get('symbol_pair2link_function', False)
        self.aggregated_kl = kwargs.get('aggregated_kl', False)
        self.n_symbols = n_symbols

        self.softplus = torch.nn.Softplus()
        offset = 0 if self.diag_in_adj_matrix else 1
        self.triu_indices = torch.triu_indices(row=n_symbols, col=n_symbols, offset=offset)
        self.mask_triu_elems = (torch.triu(torch.ones(n_symbols, n_symbols), diagonal=offset) == 1)

        if dims is not None:
            input_dim = 2 * emb_input_dim if self.symbol_pair2link_function else emb_input_dim
            self.layers = self._build_layers(input_dim, dims, nonlinearity, normalization)
            last_dim = dims[-1]
        else:
            self.layers = None
            last_dim = emb_input_dim


        output_emb_dim = 1 if self.symbol_pair2link_function else self.n_communities
        self.get_graph_embs = nn.Linear(last_dim, output_emb_dim)

    def forward(self, symbols, tau, hard=True, greedy_sampling=False):
        """
        input: symbols [n_symbols, latent_dim]
        returns symmetric matrix with Bernoulli weight.
                    Shape [n_symbols, n_symbols]
        """

        if self.symbol_pair2link_function:
            symbols = torch.cat((symbols[self.triu_indices[0]],
                                 symbols[self.triu_indices[1]]), dim=1)  # [n*(n+/-1)/2, 2*symbol_dim]

        symbols = self.layers(symbols) if self.layers is not None else symbols  # [n_symbols  or n*(n+/-1)/2, hidden_dim]

        graph_emb = self.get_graph_embs(symbols)
        if self.symbol_pair2link_function:
            bernoulli_link_logits = torch.zeros((self.n_symbols, self.n_symbols), device=self.device)
            bernoulli_link_logits[self.triu_indices[0], self.triu_indices[1]] = graph_emb.squeeze(1)
            bernoulli_link_logits = bernoulli_link_logits + torch.transpose(bernoulli_link_logits, 0, 1) \
                                    * (1.0 - torch.eye(self.n_symbols, device=self.device))
        else:
            bernoulli_link_logits = torch.tensordot(graph_emb, graph_emb, dims=[[1], [1]])  # [n_symbols, n_symbols]
        bernoulli_link_prob = torch.sigmoid(bernoulli_link_logits)
        params_graph_model = bernoulli_link_prob

        adj_matrix, link_prob = self.sample_matrix(params_graph_model, tau,
                                                            hard=hard, greedy=greedy_sampling)

        return adj_matrix, link_prob, params_graph_model

    def sample_matrix(self, params_graph_model, tau, hard=True, greedy=False):
        """
        returns adjacency matrix and link probabilities
        """

        bernoulli_link_prob = params_graph_model

        n_symbols = bernoulli_link_prob.shape[0]

        # Sample Bernoulli variable
        link_prob = bernoulli_link_prob[self.mask_triu_elems].view(-1, 1)
        link_prob = torch.cat((link_prob, 1 - link_prob), dim=1)  # [n_symbols*(n_symbols-1)/2, 2]

        s = gumbel_softmax(link_prob, tau, self.device, hard=hard)[:, 0] if not greedy \
            else greedy_sample_categorical(link_prob)[:, 0]

        adjacency = torch.zeros(n_symbols, n_symbols, device=self.device)
        adjacency[self.triu_indices[0], self.triu_indices[1]] = s
        adjacency = adjacency + torch.transpose(adjacency, 0, 1) * (1.0 - torch.eye(n_symbols, device=self.device))

        return adjacency.float(), link_prob

    def get_kl(self, link_prob, params_prior, batch_size):

        link_prob_prior, default_shape, default_rate = params_prior

        if self.aggregated_kl:
            link_prob_prior = torch.mean(link_prob_prior, dim=0)
            link_prob = torch.mean(link_prob, dim=0)

        # posterior log link probs:
        cond = (link_prob > 0.0).float()
        epsilon = torch.ones(cond.shape).fill_(1e-8).to(self.device)
        log_link_prob = torch.log(torch.max(link_prob + (1 - cond), epsilon))

        # prior log link probs:
        cond = (link_prob_prior > 0.0).float()
        epsilon = torch.ones(cond.shape).fill_(1e-8).to(self.device)
        log_link_prob_prior = torch.log(torch.max(link_prob_prior + (1 - cond), epsilon))

        kl_graph = torch.sum(link_prob * (log_link_prob - log_link_prob_prior))

        # Normalization
        kl_graph = kl_graph / float(batch_size)

        return kl_graph

    @staticmethod
    def _build_layers(input_dim: int, layers: list,
                      activation, layer_normalization: bool) -> nn.Sequential:

        activation_fn = get_class_nonlinearity(activation)
        dim = list(map(int, layers))
        n_layers = len(dim)
        layers = nn.ModuleList([])
        layers.append(nn.Linear(input_dim, dim[0]))
        if layer_normalization:
            layers.append(nn.LayerNorm(dim[0]))
        layers.append(activation_fn())
        for i in range(n_layers - 1):
            layers.append(nn.Linear(dim[i], dim[i + 1]))
            if layer_normalization:
                layers.append(nn.LayerNorm(dim[i + 1]))
            layers.append(activation_fn())

        return nn.Sequential(*layers)

    @property
    def device(self):
        return next(self.parameters()).device
