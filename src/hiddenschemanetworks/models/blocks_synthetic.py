import math
import torch as torch
from torch import nn as nn
from hiddenschemanetworks.utils.helper import gumbel_softmax
from hiddenschemanetworks.models.blocks import Block, PositionalEncoding

from hiddenschemanetworks.models.blocks import EncodeOntoRW


class SimpleTransformerEncoder(EncodeOntoRW):
    def __init__(self, vocab, fix_len, latent_dim, n_symbols,
                 voc_dim=None, emb_dim=None, **kwargs):
        super(SimpleTransformerEncoder, self).__init__(vocab, fix_len, latent_dim, n_symbols, recurrent=False, voc_dim=voc_dim,
                                                       emb_dim=emb_dim, **kwargs)


        n_heads = kwargs.get('n_heads')        # Note: head_dim = emb_dim/n_heads
        n_layers = kwargs.get('n_layers')
        dropout = kwargs.get('dropout')

        self.positional_enc = PositionalEncoding(self.emb_dim, device=self.device)

        # Transformer encoder --- embedding layers:

        hidden_dim_ = self.emb_dim
        encoder_layers = torch.nn.TransformerEncoderLayer(self.emb_dim, n_heads, hidden_dim_, dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, n_layers)
        self.get_logits = nn.Linear(self.emb_dim, n_symbols)

    def get_functions_over_nodes(self, input):
        """
        input: input data (x): [B, T]
        """

        # embed input
        x, _ = input
        x = self.embedding(x) * math.sqrt(self.emb_dim)  # [B, T, D]
        batch_size, t_len, _ = x.shape

        x = self.positional_enc(x)
        x = torch.transpose(x, 1, 0)     # [T, B, D]
        h = self.transformer_encoder(x)  # [T, B, D]
        h = self.get_logits(h.view(-1, self.emb_dim)).view(t_len, batch_size, -1)
        h = torch.transpose(h, 0, 1)     # [B, T, D]

        # walks starting points:
        f0 = nn.functional.softmax(h[:, 0], dim=-1)

        # get sentence-dependent node attributes:
        f = torch.exp(h[:, 1:])  # [B, L-1, n_symbols]

        return f0, f

class Decoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(Decoder, self).__init__()

        self.get_output = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, z):
        """
        z: [B, T, D]
        """
        output = self.get_output(z.view(-1, z.shape(2)))
        return output


class NonAutoRegEncoderRNN(Block):

    def __init__(self, vocab, fix_len, n_symbols, rw_length, voc_dim=None, emb_dim=None, **kwargs):
        super(NonAutoRegEncoderRNN, self).__init__(vocab, fix_len, None,
                                                   recurrent=True, voc_dim=voc_dim, emb_dim=emb_dim, get_embeddings=True,
                                                   **kwargs)

        dropout = kwargs.get('embedding_dropout', 0)
        self.embedding_dropout_layer = torch.nn.Dropout(p=dropout) if dropout > 0 else None

        self.rnn_dim = kwargs.get('rnn_dim')  # hidden dim in feedforward component
        self.hidden_state = None
        self.rnn_cell = nn.LSTM(input_size=self.emb_dim,
                                hidden_size=self.rnn_dim,
                                batch_first=True)
        self.hidden_state = None
        self.get_logits = nn.Linear(self.rnn_dim, n_symbols)
        self.n_symbols = n_symbols

        self.rw_length = rw_length

    def forward(self, input: tuple, tau: float, hard: bool=True):
        x, l = input
        x = self.embedding(x)
        x = self.embedding_dropout_layer(x) if self.embedding_dropout_layer is not None else x

        x = torch.nn.utils.rnn.pack_padded_sequence(x, l, True, False)
        output, self.hidden_state = self.rnn_cell(x, self.hidden_state)  # [B, T, D]

        output = torch.nn.utils.rnn.pad_packed_sequence(output, True, padding_value=self.PAD)[0]

        logits = self.get_logits(output)  # [B, T, n_symbols]
        prob = nn.functional.softmax(logits, dim=-1)

        z = gumbel_softmax(prob.view(-1, self.n_symbols), tau, self.device, hard=hard)  # [B * L, n_symbols]
        z = z.view(-1, self.rw_length, self.n_symbols)

        return z

    def initialize_hidden_state(self, batch_size: int, device: any):
        if self.recurrent:
            h = torch.zeros(1, batch_size, self.rnn_dim)
            c = torch.zeros(1, batch_size, self.rnn_dim)
            self.hidden_state = (h.to(device), c.to(device))

    def reset_history(self):
        if self.recurrent:
            self.hidden_state = tuple(x.detach() for x in self.hidden_state)
