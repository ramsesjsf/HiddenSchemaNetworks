from torch import nn
from hiddenschemanetworks.models.blocks import Block, EncodeOntoRW, PositionalEncoding
from transformers.models.gpt2.modeling_gpt2 import *
from transformers import BertModel, BertConfig

class CrossAttentionWithLearnableQueries(nn.TransformerEncoderLayer):
    def __init__(self, emb_dim, n_heads, hidden_dim, dropout):
        super(CrossAttentionWithLearnableQueries, self).__init__(emb_dim, n_heads,
                                                   dim_feedforward=hidden_dim, dropout=dropout, activation="gelu")

    def forward(self, query, keys, values, src_mask=None, src_key_padding_mask=None):
        """Pass the input through the encoder layer.
        Args:
            query:  [target_seq_len, B, D]
            keys:   [source_seq_len, B, D]
            values: [source_seq_len, B, D]
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        """
        src2 = self.self_attn(query, keys, values,
                              attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class EncoderSchema(EncodeOntoRW):
    def __init__(self, fix_len, n_symbols, **kwargs):
        super(EncoderSchema, self).__init__(None,
                                            fix_len,
                                            None,
                                            n_symbols,
                                            recurrent=False,
                                            use_huggingface_models=True,
                                            **kwargs)

        config = BertConfig().from_pretrained('bert-base-uncased')
        config.hidden_size = kwargs.get('hidden_size', config.hidden_size)
        config.num_hidden_layers = kwargs.get('num_hidden_layers', config.num_hidden_layers)
        config.num_attention_heads = kwargs.get('num_attention_heads', config.num_attention_heads)
        config.intermediate_size = kwargs.get('intermediate_size', config.intermediate_size)
        vocab_size = kwargs.get('vocab_size', None)
        if vocab_size is not None:
            config.vocab_size = vocab_size

        if kwargs.get('pretrained', True):
            self.get_hidden_states = BertModel.from_pretrained('bert-base-uncased', config=config)
        else:
            self.get_hidden_states = BertModel(config)

        self.cross_att_learn_queries = CrossAttentionWithLearnableQueries(config.hidden_size,
                                                                          config.num_attention_heads,
                                                                          config.hidden_size,
                                                                          config.hidden_dropout_prob)

        self.get_logits = nn.Linear(config.hidden_size, n_symbols)

        # Learnable queries
        self.queries = nn.Parameter(torch.randn(self.rw_length, config.hidden_size))

        self.pos_encoding_queries = PositionalEncoding(config.hidden_size, dropout=0.0)

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


class PseudoSelfAttention(GPT2Attention):
    def __init__(self, config, rw_len, symbol_dim, rw_pos_encoding=False):
        super(PseudoSelfAttention, self).__init__(config)

        self.rw_len = rw_len
        self.rw_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
        self.map_symbols = nn.Linear(symbol_dim, self.embed_dim)
        self.rw_pos_encoding = PositionalEncoding(self.embed_dim, dropout=0.0) if rw_pos_encoding else lambda z : z

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        if self.scale_attn_weights:
            attn_weights = attn_weights / (float(value.size(-1)) ** 0.5)

        if not self.is_cross_attention:
            # if only "normal" attention layer implements causal mask
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length].bool()
            attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            ###### Modify mask to include the seq of symbols in Key dim
            ###### note attn_weights: [B, n_heads, L, L + rw_len], attention_mask: [B, 1, 1, L]
            zeros = torch.zeros(attention_mask.size()[:-1] + (self.rw_len,),
                                device=attention_mask.device,
                                dtype=attention_mask.dtype)
            attention_mask = torch.cat((zeros, attention_mask), dim=-1)
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(self,
            hidden_states,
            encoder_hidden_states=None,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            use_cache=False,
            encoder_attention_mask=None,
            output_attentions=False,
    ):

        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            present = (key, value)
        else:
            present = None

        ### Compute additional key and values for rws:
        symbol_seq = self.map_symbols(encoder_hidden_states)
        symbol_seq = self.rw_pos_encoding(symbol_seq)
        key_rw, value_rw = self.rw_attn(symbol_seq).split(self.split_size, dim=2)
        key_rw = self._split_heads(key_rw, self.num_heads, self.head_dim)
        value_rw = self._split_heads(value_rw, self.num_heads, self.head_dim)
        key = torch.cat((key_rw, key), dim=-2)
        value = torch.cat((value_rw, value), dim=-2)

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)


class PseudoSelfAttentionBlock(GPT2Block):
    def __init__(self, config, rw_len, symbol_dim, rw_pos_encoding=False):
        super(GPT2Block, self).__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = PseudoSelfAttention(config, rw_len, symbol_dim, rw_pos_encoding=rw_pos_encoding)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(inner_dim, config)

    def forward(self,
            hidden_states,
            layer_past=None,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            output_attentions=False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class PseudoSelfAttentionGPT2Model(GPT2Model):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

    def __init__(self, config, rw_len, symbol_dim, rw_pos_encoding=False):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([PseudoSelfAttentionBlock(config, rw_len, symbol_dim,
                                                         rw_pos_encoding=rw_pos_encoding)
                                for _ in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None


class PseudoSelfAttentionGPT2LMHeadModel(GPT2LMHeadModel):
    _keys_to_ignore_on_load_missing = [r"attn.masked_bias", r"attn.bias", r"lm_head.weight"]

    def __init__(self, config, rw_len, symbol_dim, rw_pos_encoding=False):
        super().__init__(config)

        self.transformer = PseudoSelfAttentionGPT2Model(config, rw_len, symbol_dim, rw_pos_encoding=rw_pos_encoding)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None


class DecoderSchema(Block):
    def __init__(self, fix_len, symbol_dim, rw_len, **kwargs):
        super(DecoderSchema, self).__init__(fix_len=fix_len, use_huggingface_models=True, **kwargs)

        config = GPT2Config.from_pretrained('gpt2')
        config.n_embd = kwargs.get('hidden_size', config.n_embd)
        config.n_layer = kwargs.get('num_hidden_layers', config.n_layer)
        config.n_head = kwargs.get('num_attention_heads', config.n_head)
        config.n_inner = kwargs.get('intermediate_size', config.n_inner)
        vocab_size = kwargs.get('vocab_size', None)
        if vocab_size is not None:
            config.vocab_size = vocab_size
        rw_pos_encoding = kwargs.get('rw_pos_encoding', True)


        self.decoder_type = kwargs.get('decoder_type')
        if self.decoder_type == 'GPT2-CrossAttention':
            config.add_cross_attention = True
            self.map_symbols = nn.Linear(symbol_dim, config.hidden_size) if symbol_dim != config.hidden_size else None
            if kwargs.get('pretrained', True):
                self.get_logits, self.loading_info = GPT2LMHeadModel.from_pretrained('gpt2',
                                                                                 config=config,
                                                                                 output_loading_info=True)
            else:
                self.get_logits = GPT2LMHeadModel(config=config)

        elif self.decoder_type == 'GPT2-PseudoSelfAttention':
            if kwargs.get('pretrained', True):
                self.get_logits, self.loading_info = PseudoSelfAttentionGPT2LMHeadModel.from_pretrained('gpt2',
                                                                                                        rw_len=rw_len,
                                                                                                        symbol_dim=symbol_dim,
                                                                                                        rw_pos_encoding=True,
                                                                                                        output_loading_info=True,
                                                                                                        config=config)
            else:
                self.get_logits = PseudoSelfAttentionGPT2LMHeadModel(rw_len=rw_len,
                                                                     symbol_dim=symbol_dim,
                                                                     rw_pos_encoding=rw_pos_encoding,
                                                                     config=config)
        else:
            raise ValueError(
                "decoder_type undefined. Please choose either GPT2-CrossAttention or GPT2-PseudoSelfAttention."
            )
    def forward(self,
                input_ids,
                symbol_seq,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None, ):

        if self.decoder_type == 'GPT2-CrossAttention':
            symbol_seq = self.map_symbols(symbol_seq) if self.map_symbols is not None else symbol_seq

        logits = self.get_logits(input_ids=input_ids,
                                 encoder_hidden_states=symbol_seq,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 position_ids=position_ids,
                                 head_mask=head_mask,
                                 use_cache=False,
                                 output_attentions=False,
                                 output_hidden_states=False,
                                 return_dict=False, )[0]

        return logits

