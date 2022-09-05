from hiddenschemanetworks.models.blocks_transformers import CrossAttentionWithLearnableQueries
from hiddenschemanetworks.models.blocks import Block, PositionalEncoding
import torch
from torch import nn
from transformers import BertConfig, BertModel, GPT2Config, GPT2Model

from hiddenschemanetworks.utils.helper import greedy_sample_categorical, gumbel_softmax


class EncodeOntoRW(Block):
    def __init__(self, vocab, fix_len, latent_dim, n_symbols, aggregated_post, max_entropy_prior,
                 recurrent, voc_dim=None, emb_dim=None, use_huggingface_models=False, **kwargs):
        super(EncodeOntoRW, self).__init__(vocab, fix_len, latent_dim,
                                           recurrent=recurrent, voc_dim=voc_dim, emb_dim=emb_dim,
                                           use_huggingface_models=use_huggingface_models, **kwargs)


        self.rw_length = kwargs.get("rw_length")
        self.fully_connected_graph = kwargs.get("fully_connected_graph")
        self.const_lower_value_f_matrix = kwargs.get("const_lower_value_f_matrix", 0.0)
        self.aggregated_post = aggregated_post
        self.max_entropy_prior = max_entropy_prior
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

        if self.fully_connected_graph:
            adj_matrix = torch.ones_like(adj_matrix)

        # (Unnormalized) Posterior transition prob matrix
        f_matrix = torch.einsum('bli,blj->blij', f, f)  # Batched outer product: [B, L-1, n_symbols, n_symbols]

        f_matrix = f_matrix + self.const_lower_value_f_matrix * torch.ones_like(f_matrix)

        if adj_matrix.dim() == 3:
            q_matrix = adj_matrix.unsqueeze(1) * f_matrix  # [B, L-1, n_symbols, n_symbols]
        else:
            q_matrix = adj_matrix.view(1, 1, self.n_symbols,
                                       self.n_symbols) * f_matrix  # [B, L-1, n_symbols, n_symbols]

        if self.aggregated_post:
            walks, walk_prob, q_matrix = self.sample_agregated_walks(q_matrix, f0, tau, z_aux, hard=hard)
        else:
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
        log_q_matrix = torch.log(torch.max(q_matrix - (1 - cond), epsilon))

        # prior log transition prob:
        if self.aggregated_post and (p_matrix.dim() == 4):
            p_matrix = torch.mean(p_matrix, 0)     # [L-1, n_symbols, n_symbols]
            f0_prior = torch.mean(f0_prior, 0)     # [n_symbols]

        cond = (p_matrix > 0.0).float()
        epsilon = torch.ones(p_matrix.shape).fill_(1e-8).to(self.device)
        log_p_matrix = torch.log(torch.max(p_matrix - (1 - cond), epsilon))

        if self.aggregated_post:
            # kl starting point:
            rho_0 = walk_prob[0]  # [n_symbols]
            cond = (rho_0 > 0.0).float()
            epsilon = torch.ones(rho_0.shape).fill_(1e-8).to(self.device)
            log_rho_0 = torch.log(torch.max(rho_0 - (1 - cond), epsilon))
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


        else:
            # kl starting point:
            batch_size = walk_prob.size(0)
            rho_0 = walk_prob[:, 0]  # [B, n_symbols]
            cond = (rho_0 > 0.0).float()
            epsilon = torch.ones(rho_0.shape).fill_(1e-8).to(self.device)
            log_rho_0 = torch.log(torch.max(rho_0 - (1 - cond), epsilon))
            if f0_prior is not None:
                log_pi_0 = torch.log(torch.softmax(f0_prior, -1))
                if f0_prior.dim() == 2:
                    kl_0 = torch.sum(rho_0 * (log_rho_0 - log_pi_0))
                else:
                    kl_0 = torch.sum(rho_0 * (log_rho_0 - log_pi_0.view(1, self.n_symbols)))
            else:
                kl_0 = torch.sum(rho_0 * log_rho_0) + batch_size * log_K

            # kl full walks:
            weight = walk_prob[:, :-1].view(batch_size, -1, self.n_symbols, 1) * q_matrix  # [B, L-1, n_symbols, n_symbols]
            if log_p_matrix.dim() == 4:
                kl_rws = torch.sum(weight * (log_q_matrix - log_p_matrix))
            elif log_p_matrix.dim() == 3:
                kl_rws = torch.sum(weight * (log_q_matrix - log_p_matrix.unsqueeze(0)))
            else:
                kl_rws = torch.sum(weight * (log_q_matrix - log_p_matrix.view(1, 1, self.n_symbols, self.n_symbols)))

        # normalization
        kl_rws = kl_rws / float(batch_size) if not self.aggregated_post else kl_rws
        kl_0 = kl_0 / float(batch_size) if not self.aggregated_post else kl_0

        return kl_rws, kl_0

    def sample_walks(self, q_matrix, f0, tau, z_aux=None, hard=True, greedy=False):
        """
        q_matrix: [B, L, n_symbols, n_symbols]
        f0: [B, n_symbols]
        tau: int
        """

        q_matrix = self.get_transition_prob_matrix(q_matrix, True)

        # sample first step:
        z = gumbel_softmax(f0, tau, self.device, hard=hard) if not greedy \
            else greedy_sample_categorical(f0)      # [B, n_symbols]

        walks = torch.unsqueeze(z, 1)  # [B, 1, n_symbols]

        # get prob over walks:
        walk_prob = torch.unsqueeze(f0, 1)  # [B, 1, n_symbols]
        walk_prob_aux = None
        # random walks:
        for i in range(1, self.rw_length):

            # transition prob:
            transition_prob = torch.matmul(z.unsqueeze(1), q_matrix[:, i - 1, :]).squeeze(1)  # [B, n_symbols]

            # (*) sample step
            z = gumbel_softmax(transition_prob, tau, self.device, hard=hard) if not greedy \
                else greedy_sample_categorical(transition_prob)

            walks = torch.cat([walks, torch.unsqueeze(z, 1)], dim=1)

            # MARGINAL prob over walks (for regularization)
            f0 = torch.matmul(f0.unsqueeze(1), q_matrix[:, i - 1, :]).squeeze(1)  # [B, n_symbols]
            walk_prob = torch.cat([walk_prob, torch.unsqueeze(f0, 1)], dim=1)

            if z_aux is not None:
                if i == 1:
                    walk_prob_aux = walk_prob[:, 0, :].view(-1, 1, self.n_symbols)
                walk_prob_aux = self._get_auxiliar_walk_prob(i, z_aux, walk_prob_aux, q_matrix[:, i - 1, :])


        walk_prob = (walk_prob, walk_prob_aux)

        return walks, walk_prob, q_matrix


    def sample_agregated_walks(self, q_matrix, f0, tau, z_aux=None, hard=True, greedy=False):
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


    def initialize_hidden_state(self, batch_size: int, device: any):
        if self.recurrent:
            h = torch.zeros(2, batch_size, self.rnn_dim)
            c = torch.zeros(2, batch_size, self.rnn_dim)
            self.hidden_state = (h.to(device), c.to(device))


    def reset_history(self):
        if self.recurrent:
            self.hidden_state = tuple(x.detach() for x in self.hidden_state)


    def get_simple_kl_for_vae_mi(self, input, adj_matrix, tau, prior_params, z_aux=None, hard=True):
        """
        Returns the random walk KL for VAE-MI WITHOUT the MI term
        (that is, the standard KL as computed when self.aggregated_post is False)
        as needed when computing the PPL
        """

        p_matrix, f0_prior = prior_params
        f0, f = self.get_functions_over_nodes(input)       # [B, n_symbols], [B, L-1, n_symbols]
        f_matrix = torch.einsum('bli,blj->blij', f, f)     # [B, L-1, n_symbols, n_symbols]
        f_matrix = f_matrix + self.const_lower_value_f_matrix * torch.ones_like(f_matrix)
        if adj_matrix.dim() == 3:
            q_matrix = adj_matrix.unsqueeze(1) * f_matrix  # [B, L-1, n_symbols, n_symbols]
        else:
            q_matrix = adj_matrix.view(1, 1, self.n_symbols,
                                       self.n_symbols) * f_matrix  # [B, L-1, n_symbols, n_symbols]

        # get walk probabilities (regardless of whether self.aggregated_post is True)
        _, walk_prob, q_matrix = self.sample_walks(q_matrix, f0, tau, z_aux, hard=hard)
        walk_prob, _ = walk_prob

        # compute the KL
        log_K = torch.log(torch.tensor(self.n_symbols).to(self.device).float())

        # posterior log transition prob:
        cond = (q_matrix > 0.0).float()
        epsilon = torch.ones(q_matrix.shape).fill_(1e-8).to(self.device)
        log_q_matrix = torch.log(torch.max(q_matrix + (1 - cond), epsilon))

        # prior log transition prob:
        cond = (p_matrix > 0.0).float()
        epsilon = torch.ones(p_matrix.shape).fill_(1e-8).to(self.device)
        log_p_matrix = torch.log(torch.max(p_matrix + (1 - cond), epsilon))

        # kl starting point:
        batch_size = walk_prob.size(0)
        rho_0 = walk_prob[:, 0]  # [B, n_symbols]
        cond = (rho_0 > 0.0).float()
        epsilon = torch.ones(rho_0.shape).fill_(1e-8).to(self.device)
        log_rho_0 = torch.log(torch.max(rho_0 + (1 - cond), epsilon))
        if f0_prior is not None:
            log_pi_0 = torch.log(torch.softmax(f0_prior, 0))
            kl_0 = torch.sum(rho_0 * (log_rho_0 - log_pi_0.view(1, self.n_symbols)))
        else:
            kl_0 = torch.sum(rho_0 * log_rho_0) + batch_size * log_K

        # kl full walks:
        weight = walk_prob[:, :-1].view(batch_size, -1, self.n_symbols, 1) * q_matrix  # [B, L-1, n_symbols, n_symbols]
        kl_rws = torch.sum(weight * (log_q_matrix - log_p_matrix.view(1, -1, self.n_symbols, self.n_symbols)))

        # normalization
        kl_rws = kl_rws / float(batch_size)
        kl_0 = kl_0 / float(batch_size)

        return kl_rws, kl_0


    @property
    def output_size(self):
        return self.__output_size


class EncoderSchema(EncodeOntoRW):
    def __init__(self, fix_len, n_symbols, aggregated_post, max_entropy_prior, **kwargs):
        super(EncoderSchema, self).__init__(None,
                                            fix_len,
                                            None,
                                            n_symbols,
                                            aggregated_post,
                                            max_entropy_prior,
                                            False,
                                            use_huggingface_models=True,
                                            **kwargs)

        self.encoder_type = kwargs.get('encoder_type')
        pretrained = kwargs.get('pretrained', True)
        if self.encoder_type == 'BERT':
            self.config = BertConfig().from_pretrained('bert-base-uncased')
            if not pretrained:
                self.config.hidden_size = kwargs.get('hidden_size', self.config.hidden_size)
                self.config.num_hidden_layers = kwargs.get('num_hidden_layers', self.config.num_hidden_layers)
                self.config.num_attention_heads = kwargs.get('num_attention_heads', self.config.num_attention_heads)
                self.config.intermediate_size = kwargs.get('intermediate_size', self.config.intermediate_size)
            vocab_size = kwargs.get('vocab_size', None)
            if vocab_size is not None:
                self.config.vocab_size = vocab_size
            if pretrained:
                    self.get_hidden_states = BertModel.from_pretrained('bert-base-uncased', config=self.config,
                                                                       ignore_mismatched_sizes=True)
            else:
                self.get_hidden_states = BertModel(self.config)
        else:
            raise ValueError(
                "encoder_type undefined. Please choose either BERT or GPT2."
            )

        self.cross_att_learn_queries = CrossAttentionWithLearnableQueries(self.config.hidden_size,
                                                                          self.config.num_attention_heads,
                                                                          self.config.hidden_size,
                                                                          self.config.hidden_dropout_prob)

        self.get_logits = nn.Linear(self.config.hidden_size, n_symbols)

        # Learnable queries
        self.queries = nn.Parameter(torch.randn(self.rw_length, self.config.hidden_size))
        self.add_pos_encoding = kwargs.get('pos_encoding', False)
        if self.add_pos_encoding:
            self.pos_encoding_queries = PositionalEncoding(self.config.hidden_size, dropout=0.0)

        self.query_init()

    def get_functions_over_nodes(self, input):

        input_ids, attention_mask = input

        hidden_states = self.get_hidden_states(input_ids=input_ids,
                                               attention_mask=attention_mask,
                                               token_type_ids=None,
                                               position_ids=None,
                                               head_mask=None,
                                               use_cache=False,
                                               output_attentions=False,
                                               output_hidden_states=False,
                                               return_dict=False,)[0]    # [B, L, D]

        batch_size, _, emb_dim = hidden_states.shape

        # queries to change from seq_len to rw_len
        q = self.queries    # [target_len, D]
        q = q.unsqueeze(0).repeat((batch_size, 1, 1))
        q = torch.transpose(q, 1, 0)  # [target_len, B, D]
        hidden_states = torch.transpose(hidden_states, 1, 0)  # [L, B, D]
        hidden_states = self.cross_att_learn_queries(q, hidden_states, hidden_states)  # [target_len, B, D]
        hidden_states = torch.transpose(hidden_states, 1, 0)  # [B, target_len, D]

        logits = self.get_logits(hidden_states)

        # walks starting points:
        f0 = nn.functional.softmax(logits[:, 0], dim=-1)

        # get sentence-dependent node attributes:
        f = torch.exp(logits[:, 1:])  # [B, target_len-1, n_symbols]

        return f0, f

    def query_init(self):
        torch.nn.init.xavier_normal_(self.queries)


class EncoderAtomic2(EncoderSchema):
    """
    Has the special prior that is just the sub+rel rw half
    """

    def __init__(self, fix_len, n_symbols, aggregated_post, max_entropy_prior, **kwargs):
        super(EncoderAtomic2, self).__init__(fix_len, n_symbols, aggregated_post, max_entropy_prior, **kwargs)

        n_layers_prior = kwargs.get('n_layers_prior')
        prior_transformer_layers = torch.nn.TransformerEncoderLayer(self.config.hidden_size, self.config.num_attention_heads,
                                                                    self.config.hidden_size, self.config.hidden_dropout_prob)
        self.prior_transformer_blocks = torch.nn.TransformerEncoder(prior_transformer_layers,
                                                                    n_layers_prior)

        self.cross_att_learn_queries_prior = self.cross_att_learn_queries

        self.cross_att_learn_queries_post = CrossAttentionWithLearnableQueries(self.config.hidden_size,
                                                                               self.config.num_attention_heads,
                                                                               self.config.hidden_size,
                                                                               self.config.hidden_dropout_prob)

        self.get_logits_prior = nn.Linear(self.config.hidden_size, n_symbols)
        self.get_logits_post = nn.Linear(self.config.hidden_size, n_symbols)

        # Learnable queries
        self.queries_prior = self.queries
        self.queries_post = nn.Parameter(torch.randn(self.rw_length, self.config.hidden_size))

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

        f0_prior, f_prior, f0_post, f_post = self.get_functions_over_nodes(input)  # [B, n_symbols], [B, L-1, n_symbols]

        if self.fully_connected_graph:
            adj_matrix = torch.ones_like(adj_matrix)

        # (Unnormalized) Posterior transition prob matrix
        f_matrix = torch.einsum('bli,blj->blij', f_post,
                                f_post)  # Batched outer product: [B, L-1, n_symbols, n_symbols]
        f_matrix = f_matrix + self.const_lower_value_f_matrix * torch.ones_like(f_matrix)

        p_matrix = torch.einsum('bli,blj->blij', f_prior,
                                f_prior)  # Batched outer product: [B, L-1, n_symbols, n_symbols]

        if adj_matrix.dim() == 3:
            q_matrix = adj_matrix.unsqueeze(1) * f_matrix  # [B, L-1, n_symbols, n_symbols]
            p_matrix = adj_matrix.unsqueeze(1) * p_matrix  # [B, L-1, n_symbols, n_symbols]
        else:
            q_matrix = adj_matrix.view(1, 1, self.n_symbols,
                                       self.n_symbols) * f_matrix  # [B, L-1, n_symbols, n_symbols]
            p_matrix = adj_matrix.view(1, 1, self.n_symbols,
                                       self.n_symbols) * p_matrix  # [B, L-1, n_symbols, n_symbols]

        p_matrix = self.get_transition_prob_matrix(p_matrix, batched=True)

        if self.aggregated_post:
            walks, walk_prob, q_matrix = self.sample_agregated_walks(q_matrix, f0_post, tau, z_aux, hard=hard)
        else:
            walks, walk_prob, q_matrix = self.sample_walks(q_matrix, f0_post, tau, z_aux, hard=hard)

        walk_prob, walk_prob_aux = walk_prob

        kl_rws, kl_0 = self.get_kl_rws(walk_prob, q_matrix, p_matrix, f0_prior)

        return walks, kl_rws, kl_0, walk_prob_aux

    def get_functions_over_nodes(self, input):
        """
        (input_ids, (attention_mask, token_type_ids)) = input
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        Mask values selected in [0, 1]:
            1 for tokens that are not masked,
            0 for tokens that are masked.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        Indices are selected in [0, 1]:
            0 corresponds to a sentence A token,
            1 corresponds to a sentence B token.
        """

        input_ids, masks = input
        attention_mask, token_type_ids = masks

        # get contextual embeddings [h_s] for (subject + relation)
        attention_mask_object = torch.logical_not(token_type_ids) * attention_mask

        # mask properly by replacing object ids with padding ids
        input_ids_masked = input_ids.clone()
        input_ids_masked[attention_mask_object == 0] = 0

        h_prior = self.get_hidden_states(input_ids=input_ids_masked,
                                         attention_mask=attention_mask_object,
                                         token_type_ids=None,
                                         position_ids=None,
                                         head_mask=None,
                                         use_cache=False,
                                         output_attentions=False,
                                         output_hidden_states=False,
                                         return_dict=False, )[0]  # [B, L, D]

        h_post = self.get_hidden_states(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=None,
                                        position_ids=None,
                                        head_mask=None,
                                        use_cache=False,
                                        output_attentions=False,
                                        output_hidden_states=False,
                                        return_dict=False, )[0]  # [B, L, D]

        batch_size, _, emb_dim = h_prior.shape

        # prior model:
        q_prior = self.queries_prior  # [target_len, D]
        q_prior = q_prior.unsqueeze(0).repeat((batch_size, 1, 1))
        if self.add_pos_encoding:
            q_prior = self.pos_encoding_queries(q_prior)
        q_prior = torch.transpose(q_prior, 1, 0)  # [target_len, B, D]

        h_prior = torch.transpose(h_prior, 1, 0)  # [L, B, D]
        h_prior = self.cross_att_learn_queries_prior(q_prior, h_prior, h_prior)  # [target_len/2, B, D]
        h_prior = self.prior_transformer_blocks(h_prior)
        h_prior = torch.transpose(h_prior, 1, 0)  # [B, target_len/2, D]

        # posterior model:
        q_post = self.queries_post  # [target_len, D]
        q_post = q_post.unsqueeze(0).repeat((batch_size, 1, 1))
        if self.add_pos_encoding:
            q_post = self.pos_encoding_queries(q_post)
        q_post = torch.transpose(q_post, 1, 0)  # [target_len, B, D]

        h_post = torch.transpose(h_post, 1, 0)  # [L, B, D]
        h_post = self.cross_att_learn_queries_post(q_post, h_post, h_post)  # [target_len/2, B, D]
        h_post = torch.transpose(h_post, 1, 0)  # [B, target_len/2, D]

        logits_prior = self.get_logits_prior(h_prior)
        f0_prior = nn.functional.softmax(logits_prior[:, 0], dim=-1)  # walks starting points:
        f_prior = torch.exp(logits_prior[:, 1:])  # sentence-dependent node attributes: [B, target_len-1, n_symbols]

        logits_post = self.get_logits_post(h_post)
        f0_post = nn.functional.softmax(logits_post[:, 0], dim=-1)  # walks starting points:
        f_post = torch.exp(logits_post[:, 1:])  # sentence-dependent node attributes: [B, target_len-1, n_symbols]

        return f0_prior, f_prior, f0_post, f_post

    def sample_conditioned_sub_rel(self, input, adj_matrix, p_matrix, tau=1, z_aux=None, hard=True, greedy=False,
                                   use_posterior_rw=False):
        """
        input: input data: tuple(input_ids, masks) where masks = tuple(attention_mask, token_type_ids)
               adj_matrix: n_symbols, n_symbols]
               p_matrix: [L-1, n_symbols, n_symbols]
               temperature Gumbel distribution (tau): int

        Returns: one-hot sequence over symbols (z): [B, L, n_symbols]
        """

        f0_prior, f_prior, _, _ = self.get_functions_over_nodes(input)  # [B, n_symbols], [B, L-1, n_symbols]

        if self.fully_connected_graph:
            adj_matrix = torch.ones_like(adj_matrix)

        # (Unnormalized) transition prob matrix
        f_matrix = torch.einsum('bli,blj->blij', f_prior,
                                f_prior)  # Batched outer product: [B, L-1, n_symbols, n_symbols]

        f_matrix = f_matrix + self.const_lower_value_f_matrix * torch.ones_like(f_matrix)

        if adj_matrix.dim() == 3:
            q_matrix = adj_matrix.unsqueeze(1) * f_matrix  # [B, L-1, n_symbols, n_symbols]
        else:
            q_matrix = adj_matrix.view(1, 1, self.n_symbols,
                                       self.n_symbols) * f_matrix  # [B, L-1, n_symbols, n_symbols]

        if self.aggregated_post:
            walks, _, _ = self.sample_agregated_walks(q_matrix, f0_prior, tau, z_aux, hard=hard, greedy=greedy)
        else:
            walks, _, _ = self.sample_walks(q_matrix, f0_prior, tau, z_aux, hard=hard, greedy=greedy)

        return walks


class EncoderAtomic3(EncoderAtomic2):
    def __init__(self, fix_len, n_symbols, aggregated_post, max_entropy_prior, **kwargs):
        super(EncoderAtomic3, self).__init__(fix_len, n_symbols, aggregated_post, max_entropy_prior, **kwargs)

        self.mask_subject_relation_from_rw_end = kwargs.get('mask_subject_relation_from_rw_end', False)

    def get_functions_over_nodes(self, input):
        """
        (input_ids, (attention_mask, token_type_ids)) = input
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        Mask values selected in [0, 1]:
            1 for tokens that are not masked,
            0 for tokens that are masked.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        Indices are selected in [0, 1]:
            0 corresponds to a sentence A token,
            1 corresponds to a sentence B token.
        """

        input_ids, masks = input
        attention_mask, token_type_ids = masks

        # get contextual embeddings [h_s] for (subject + relation)
        attention_mask_object = torch.logical_not(token_type_ids) * attention_mask

        # mask properly by replacing object ids with padding ids
        input_ids_masked = input_ids.clone()
        input_ids_masked[attention_mask_object == 0] = 0

        h_subject = self.get_hidden_states(input_ids=input_ids_masked,
                                           attention_mask=attention_mask_object,
                                           token_type_ids=None,
                                           position_ids=None,
                                           head_mask=None,
                                           use_cache=False,
                                           output_attentions=False,
                                           output_hidden_states=False,
                                           return_dict=False, )[0]  # [B, L, D]

        if self.mask_subject_relation_from_rw_end:
            attention_mask = token_type_ids

        h_object = self.get_hidden_states(input_ids=input_ids,
                                          attention_mask=attention_mask,
                                          token_type_ids=None,
                                          position_ids=None,
                                          head_mask=None,
                                          use_cache=False,
                                          output_attentions=False,
                                          output_hidden_states=False,
                                          return_dict=False, )[0]  # [B, L, D]

        batch_size, _, emb_dim = h_subject.shape

        # prior model:
        q_prior = self.queries_prior  # [target_len, D]
        q_prior = q_prior.unsqueeze(0).repeat((batch_size, 1, 1))
        if self.add_pos_encoding:
            q_prior = self.pos_encoding_queries(q_prior)
        q_prior = torch.transpose(q_prior, 1, 0)  # [target_len, B, D]

        h_subject = torch.transpose(h_subject, 1, 0)  # [L, B, D]
        h_prior = self.cross_att_learn_queries_prior(q_prior, h_subject, h_subject)  # [target_len/2, B, D]
        h_prior = self.prior_transformer_blocks(h_prior)
        h_prior = torch.transpose(h_prior, 1, 0)  # [B, target_len/2, D]

        # posterior model:
        q_post = self.queries_post  # [target_len, D]
        q_post = q_post.unsqueeze(0).repeat((batch_size, 1, 1))
        if self.add_pos_encoding:
            q_post = self.pos_encoding_queries(q_post)

        q_post = torch.transpose(q_post, 1, 0)  # [target_len, B, D]
        q_post_1 = q_post[:int(self.rw_length / 2)]  # [target_len/2, B, D]
        q_post_2 = q_post[int(self.rw_length / 2):]  # [target_len/2, B, D]

        h_post_1 = self.cross_att_learn_queries_post(q_post_1, h_subject, h_subject)  # [target_len/2, B, D]
        h_post_1 = torch.transpose(h_post_1, 1, 0)  # [B, target_len/2, D]

        h_object = torch.transpose(h_object, 1, 0)  # [L, B, D]
        h_post_2 = self.cross_att_learn_queries_post(q_post_2, h_object, h_object)  # [target_len/2, B, D]
        h_post_2 = torch.transpose(h_post_2, 1, 0)  # [B, target_len/2, D]

        h_post = torch.cat([h_post_1, h_post_2], dim=1)

        logits_prior = self.get_logits_prior(h_prior)
        f0_prior = nn.functional.softmax(logits_prior[:, 0], dim=-1)  # walks starting points:
        f_prior = torch.exp(logits_prior[:, 1:])  # sentence-dependent node attributes: [B, target_len-1, n_symbols]

        logits_post = self.get_logits_post(h_post)
        f0_post = nn.functional.softmax(logits_post[:, 0], dim=-1)  # walks starting points:
        f_post = torch.exp(logits_post[:, 1:])  # sentence-dependent node attributes: [B, target_len-1, n_symbols]

        return f0_prior, f_prior, f0_post, f_post

    def sample_conditioned_sub_rel(self, input, adj_matrix, p_matrix_, tau=1, z_aux=None, hard=True, greedy=False,
                                   use_posterior_rw=False):
        """
        input: input data: tuple(input_ids, masks) where masks = tuple(attention_mask, token_type_ids)
               adj_matrix: n_symbols, n_symbols]
               p_matrix: [L-1, n_symbols, n_symbols]
               temperature Gumbel distribution (tau): int

        Returns: one-hot sequence over symbols (z): [B, L, n_symbols]
        """

        _, f_prior, f0_post, f_post = self.get_functions_over_nodes(input)  # [B, n_symbols], [B, L-1, n_symbols]

        if self.fully_connected_graph:
            adj_matrix = torch.ones_like(adj_matrix)

        # (Unnormalized) transition prob matrix
        f_matrix = torch.einsum('bli,blj->blij', f_post,
                                f_post)  # Batched outer product: [B, L-1, n_symbols, n_symbols]
        p_matrix = torch.einsum('bli,blj->blij', f_prior, f_prior)

        f_matrix = f_matrix + self.const_lower_value_f_matrix * torch.ones_like(f_matrix)
        p_matrix = p_matrix + self.const_lower_value_f_matrix * torch.ones_like(p_matrix)

        batch_size = f_post.shape[0]
        mask = torch.cat(
            [torch.ones([batch_size, int(self.rw_length / 2) - 1]), torch.zeros([batch_size, int(self.rw_length / 2)])],
            dim=1).unsqueeze(-1).unsqueeze(-1).to(self.device)  # [B, L-1, 1, 1]

        f_matrix = f_matrix * mask + p_matrix * torch.logical_not(mask).float()

        if adj_matrix.dim() == 3:
            q_matrix = adj_matrix.unsqueeze(1) * f_matrix  # [B, L-1, n_symbols, n_symbols]
        else:
            q_matrix = adj_matrix.view(1, 1, self.n_symbols,
                                       self.n_symbols) * f_matrix  # [B, L-1, n_symbols, n_symbols]

        if self.aggregated_post:
            walks, _, _ = self.sample_agregated_walks(q_matrix, f0_post, tau, z_aux, hard=hard, greedy=greedy)
        else:
            walks, _, _ = self.sample_walks(q_matrix, f0_post, tau, z_aux, hard=hard, greedy=greedy)

        return walks