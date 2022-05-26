import logging
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from hiddenschemanetworks.utils.helper import create_instance, load_params
from hiddenschemanetworks.utils.interpolation import Interpolation
from hiddenschemanetworks.models.languagemodels import RealSchema

class Model():

    def __init__(self, path, model_name, version='best_model.pth', device='cuda:1', data_loader=None, batch_size=32,
                 custom_path_to_data=None):
        """
        load a trained RealSchemaPretrained model saved in '<path>/<model_name>/<version>
        with .yaml file '<path>/<model_name>/config.yaml'
        version defaults to 'best_model.pth'
        """
        logger = logging.getLogger()
        self.model_name = model_name
        self.params = load_params(os.path.join(path, model_name, 'config.yaml'), logger)
        self.params['gpus'] = [device[-1]]
        self.device = torch.device(device)

        if data_loader is None:
            if custom_path_to_data is not None:
                self.params['data_loader']['args']['path_to_data'] = custom_path_to_data
            self.params['data_loader']['args']['batch_size'] = batch_size
            self.data_loader = create_instance('data_loader', self.params, self.device)
        else:
            self.data_loader = data_loader

        self.model = create_instance('model',
                                     self.params,
                                     self.data_loader.pad_token_id,
                                     self.data_loader.fix_len).to(self.device)

        state_dict = torch.load(os.path.join(path + model_name, version), self.device)['model_state']

        try:
            self.model.load_state_dict(state_dict)
        except:
            print('could not load model state')

        self.model.eval()

        self.is_schema = isinstance(self.model, RealSchema)

    def data2string(self, x, prefix=None):
        """
        transforms sequences of tokens into a string of detokenized lines, each corresponding to a sequences
        x: a batch of token sequences (list of 1d tensors or 2d tensor)
        prefix: optional list of prefixes, each to be appended in front of each line
        """
        tokenizer = self.data_loader.train.dataset.tokenizer_list[-1]
        string = ''
        for i, xi in enumerate(x):
            if prefix is not None:
                string += str(prefix[i])
            sentence = torch.tensor(xi).view(-1)
            sentence = tokenizer.batch_decode(sentence, skip_special_tokens=False)
            sentence = sentence[1:] if sentence[0] == tokenizer.bos_token else sentence
            for word in sentence:
                if word == tokenizer.eos_token:
                    break
                string += word
            string += '\n'
        return string

    def generate(self, rw=None, num_samples=10, decoding_method='greedy', K=5, print_text=True):
        """
        generate num_samples text sequences
        rw (random walks): [num_samples, seq_len, n_symbols]
            if rw is None, samples random walks from prior
        num_samples: number of generated samples
        """
        if self.is_schema:
            if rw is None:
                graph = self.sample_graph()
                rw = self.model.sample_rw_prior(num_samples, graph)
            else:
                num_samples = rw.size(0)
            symbol_seq = torch.matmul(rw, self.model.symbols)
        else:
            symbol_seq = None

        sequence = torch.full((num_samples,),
                              self.data_loader.tokenizer.bos_token_id,
                              device=self.device).unsqueeze(1)


        if decoding_method != 'K-beam_search':
            sequence = self._batch_decoder(sequence, symbol_seq, 1, K, decoding_method)
        else:
            sequence = self._batched_beam_search_decoder(sequence, symbol_seq, 1, K)

        if print_text:
            seq_len = sequence.shape[-1]
            sequence = sequence.view(-1, seq_len) if decoding_method == 'K-beam_search' else sequence
            new_n_samples = sequence.shape[0]
            sequence = self.data_loader.tokenizer.batch_decode(sequence, skip_special_tokens=True)
            print('generated sentences:')
            for i in range(new_n_samples):
                print('-' * 10)
                if decoding_method == 'K-beam_search' and (i % K)==0:
                    print('K-size set')
                print(sequence[i])
            print('-' * 10)
        else:
            return sequence

    def interpolate(self, num_interpolations=5, random_walks=None):
        """
        interpolate using the shortest path between each pair of nodes in two random walks
        num_interpolations: maximum number of interpolation steps
        random_walks: random walks to be interpolated [num_samples, seq_len, n_symbols]
            if None, sample from prior
        """
        with torch.no_grad():
            graph = self.sample_graph()
            if random_walks is None:
                random_walks = self.model.sample_rw_prior(2, graph)
            try:
                interpolation = Interpolation(graph.cpu(), random_walks.cpu())
            except:
                print("random walks reside on unconnected subgraphs, so no path can be found")
                return

            random_walk_sequence = random_walks[0].unsqueeze(0)

            for pam in (torch.arange(num_interpolations) / num_interpolations):
                interpolated_walk = interpolation.interpolated_walk(pam)
                interpolated_walk = interpolation.to_onehot(interpolated_walk)
                interpolated_walk = torch.tensor(interpolated_walk).unsqueeze(0).to(self.device)
                random_walk_sequence = torch.cat((random_walk_sequence, interpolated_walk), dim=0)
            random_walk_sequence = torch.cat((random_walk_sequence, random_walks[1].unsqueeze(0)), dim=0)

            self.generate(random_walk_sequence.float())

    def recombine_at_pos(self, random_walks=None):
        """
        interpolate by cutting two random walks at every position and recombining them
        random_walks: random walks to be interpolated [num_samples, seq_len, n_symbols]
            if None, sample from prior
        """
        with torch.no_grad():
            graph = self.sample_graph()
            if random_walks is None:
                random_walks = self.model.sample_rw_prior(2, graph)
            random_walk_sequence = random_walks[0].unsqueeze(0)
            for pos in range(1, random_walks.size(1)):
                interpolated_walk = torch.cat((random_walks[0, :-pos], random_walks[1, -pos:]), dim=0)
                interpolated_walk = torch.tensor(interpolated_walk).unsqueeze(0).to(self.device)
                random_walk_sequence = torch.cat((random_walk_sequence, interpolated_walk), dim=0)

            self.generate(random_walk_sequence.float(), random_walks.size(1))

    def print_reconstructions(self, n_samples=30):
        """
        generates n_samples text sequences from random walks sampled from the posterior
        """
        rw = self.sample_rw_posterior(n_samples)
        self.generate(rw, num_samples=n_samples)

    def sample_rw_posterior(self, n_samples=2):
        """
        sample n_samples random walks from the posterior (test set)
        """
        if n_samples > self.data_loader.batch_size:
            n_samples = self.data_loader.batch_size
        minibatch = next(iter(self.data_loader.test))
        enc_input_seq = minibatch['input_enc'].long().to(self.device)[:n_samples]
        enc_attn_mask = minibatch['attn_mask_enc'].long().to(self.device)[:n_samples]
        dec_input_seq = dec_attn_mask = torch.zeros_like(enc_input_seq).to(self.device)

        rw = self.model(enc_input_seq, dec_input_seq, enc_attn_mask, dec_attn_mask, hard_graph_samples=False)[1]
        return rw

    def save_rw_graph_label(self, path_to_dir, single_graph=True):
        """
        saves the graph probs, labels and posterior random walks from test set in <path_to_dir>
        rws.pt contains random walks of node indices, shape [n_data, rw_len]
        labels.pt contains label of each sequence, shape [n_data]
        graph_probs.pt contains the edge probabilities of the graph, shape [n_symbols, n_symbols]
        if single_graph:
            graph.pt contains the sampled graph the random walks were computed with, shape [n_symbols, n_symbols]
        """
        path_to_rws = os.path.join(path_to_dir, 'rws.pt')
        path_to_lbl = os.path.join(path_to_dir, 'labels.pt')
        path_to_graph_probs = os.path.join(path_to_dir, 'graph_probs.pt')
        path_to_graph = os.path.join(path_to_dir, 'graph.pt')
        rw_list = []
        label_list = []

        if single_graph:
            adj_mat = self.sample_graph()
            self.model.adj_matrix = adj_mat

        print('computing random walks...')
        for minibatch in tqdm(self.data_loader.test):
            enc_input_seq = minibatch['input_enc'].long().to(self.device)
            enc_attn_mask = minibatch['attn_mask_enc'].long().to(self.device)
            dec_input_seq = dec_attn_mask = torch.zeros_like(enc_input_seq).to(self.device)

            _, rw, _, _, _, _ = self.model(enc_input_seq, dec_input_seq, enc_attn_mask, dec_attn_mask)

            rw = rw.argmax(dim=-1)
            rw_list.append(rw.detach().cpu())

            if 'label' in minibatch.keys():
                label = minibatch['label']
                label_list.append(label)


        print(f'saving random walks to {path_to_rws}')
        rws = torch.cat(rw_list, dim=0)
        torch.save(rws, path_to_rws)

        if label_list:
            print(f'saving labels to {path_to_lbl}')
            labels = torch.cat(label_list, dim=0)
            torch.save(labels, path_to_lbl)

        print(f'saving graph link probs to {path_to_graph_probs}')
        graph_probs = self.model.graph_generator(schema.model.symbols, 1)[2]
        torch.save(graph_probs.detach().cpu(), path_to_graph_probs)
        if single_graph:
            print(f'saving graph to {path_to_graph}')
            torch.save(adj_mat.detach().cpu(), path_to_graph)
        print('done.')

    def histogram(self):
        """
        plot a histogram of nodes visited on random walks upon inference on the test set
        compute entropy of the normalized histogram
        """
        hgram = torch.zeros((self.model.n_symbols,))
        print('computing histogram...')
        for minibatch in tqdm(self.data_loader.test):
            enc_input_seq = minibatch['input_enc'].long().to(self.device)
            dec_input_seq = minibatch['input_dec'].long().to(self.device)
            enc_attn_mask = minibatch['attn_mask_enc'].long().to(self.device)
            dec_attn_mask = minibatch['attn_mask_dec'].long().to(self.device)

            _, rw, _, _, _, graph = self.model(enc_input_seq, dec_input_seq, enc_attn_mask, dec_attn_mask)
            n_links = graph.sum(dim=0).cpu()
            n_links[n_links == 0] = 1
            symbols = torch.argmax(rw, dim=-1)
            hgram += np.histogram(symbols.cpu(), np.arange(self.model.n_symbols + 1) - .5, density=False)[0] / n_links
        plt.bar(range(len(hgram)), hgram)
        plt.show()
        pmf = hgram / torch.sum(hgram)
        distr = torch.distributions.Categorical(pmf)
        print('entropy of symbol distribution: ', distr.entropy())
        print('entropy of uniform distribution:', -torch.log(torch.tensor(1 / self.model.n_symbols)))
        print('minimum number of visits of a node:', hgram.min())

    def evaluate_metrics(self, return_values=False):
        ppl = torch.zeros(1).to(self.device)
        nll_loss = torch.zeros(1).to(self.device)
        n_hits = torch.zeros(1).to(self.device)
        if self.is_schema:
            kl_graph = torch.zeros(1).to(self.device)
            kl_rws = torch.zeros(1).to(self.device)
        print('computing metrics...')
        for i, minibatch in enumerate(self.data_loader.test):
            minibatch = {key: value.to(self.device) for key, value in minibatch.items()}
            stats = self.model.validate_step(minibatch)
            ppl = ppl + stats['PPL']
            nll_loss = nll_loss + stats['NLL-Loss']
            n_hits = n_hits + stats['number_of_hits']
            if self.is_schema:
                kl_graph = kl_graph + stats['KL-Graph']
                kl_rws = kl_rws + stats['KL-RWs'] + stats['KL-0']

        print('-' * 10)
        print('metrics:')
        print('number of hits: ', n_hits / float(len(self.data_loader.test)))
        print('PPL: ', ppl / float(len(self.data_loader.test)))
        print('NLL-Loss: ', nll_loss / float(len(self.data_loader.test)))
        if self.is_schema:
            print('KL-Graph: ', kl_graph / float(len(self.data_loader.test)))
            print('KL-RWs: ', kl_rws / float(len(self.data_loader.test)))
            print('expected number of edges in graph:', self.ecount())
        print('-' * 10)

        if return_values:
            ppl = ppl / float(len(self.data_loader.test))
            kl_graph = kl_graph / float(len(self.data_loader.test))
            kl_rws = kl_rws / float(len(self.data_loader.test))
            return ppl, kl_graph, kl_rws

    def ecount(self):
        """
        returns expectation of number of edges in the graph
        """
        link_probs = self.model.graph_generator(self.model.symbols, 1, greedy_sampling=False)[2]
        return 0.5 * torch.sum(link_probs)

    def sample_graph(self, greedy=False):
        """
        sample a graph from the graph distribution
        returns the adjacency matrix
        """
        graph = self.model.graph_generator(self.model.symbols, 1,
                                           hard=True, greedy_sampling=greedy)[0]
        return graph

    def _batched_beam_search_decoder(self, sequence, symbol_seq, length, K):
        """
        input_sequence: [B, seq_len]
        K: number of beams
        Returns sequences of shape [B, K, seq_len]
        """
        EOS = torch.full((1, K), self.data_loader.tokenizer.eos_token_id).to(self.device)  # [1, K]
        attn_mask = torch.ones_like(sequence, device=self.device)  # [B, t]

        # Decode first step:
        if self.is_schema:
            logits = self.model.decoder(sequence, symbol_seq, attn_mask)  # [B, t, vocab_size]
        else:
            logits = self.model(sequence, attn_mask)
        logits = torch.nn.functional.log_softmax(logits, -1)
        batch_size, len, voc_dim = logits.shape

        # Define vocabulary indices:
        vocab_ind = torch.arange(voc_dim).unsqueeze(0).unsqueeze(0).expand(batch_size, K, -1).unsqueeze(-1).to(self.device)  # [B, K, V, 1]

        # Kill mask
        kill_mask = torch.full((1, 1, voc_dim), -10_000).to(self.device)
        kill_mask[:, :, self.data_loader.tokenizer.eos_token_id] = 0.0   # [1, 1, vocal_size]

        # get indices of sorted logits:
        scores, indices = torch.sort(logits[:, -1], descending=True)  # [B, vocab_size]

        # top K logits:
        prediction = indices[:, :K]  # [B, K]
        scores = scores[:, :K]       # [B, K]

        # generate mask:
        mask = (prediction != EOS)  # [B, K]

        # expand input sequence into beam seq:
        sequence = sequence.unsqueeze(1).expand(-1, K, -1)             # [B, K, t]
        sequence = torch.cat((sequence,
                              prediction.unsqueeze(-1)), dim=-1)       # [B, K, t+1]
        attn_mask = attn_mask.unsqueeze(1).expand(-1, K, -1)           # [B, K, t]
        attn_mask = torch.cat([attn_mask,
                               mask.unsqueeze(-1)], dim=-1)            # [B, K, t+1]
        if self.is_schema:
            symbol_seq = symbol_seq.unsqueeze(1).expand(-1, K, -1, -1)     # [B, K, L, N_symbols]

        total_seq_len = len + 1
        for j in range(self.model.fix_len - length - 1):

            # reshape to treat the K samples as batch elements
            sequence = sequence.view(batch_size * K, -1)
            if self.is_schema:
                symbol_seq = symbol_seq.reshape(batch_size * K, self.model.rw_length, self.model.n_symbols)
            attn_mask = attn_mask.view(batch_size * K, -1)

            # get logits
            if self.is_schema:
                logits = self.model.decoder(sequence, symbol_seq, attn_mask)  # [B*K, t+1, vocab_size]
            else:
                logits = self.model(sequence, attn_mask)
            logits = torch.nn.functional.log_softmax(logits, -1)

            # reshape back
            sequence = sequence.view(batch_size, K, -1)
            if self.is_schema:
                symbol_seq = symbol_seq.view(batch_size, K, self.model.rw_length, self.model.n_symbols)
            attn_mask = attn_mask.view(batch_size, K, -1)
            logits = logits.view(batch_size, K, -1, voc_dim)

            # set of tentative sequences
            tentative_seqs = torch.cat([sequence.unsqueeze(2).expand(-1, -1, voc_dim, -1),
                                        vocab_ind], dim=-1)  # [B, K, V, t + 2 + j]

            tmp_len = tentative_seqs.shape[-1]

            # find top K sequences
            counts = attn_mask.sum(-1).unsqueeze(-1)  # [B, K, 1]

            scores = torch.where(counts < total_seq_len,
                                 scores.unsqueeze(-1) + kill_mask,
                                 scores.unsqueeze(-1) + logits[:, :, -1])   # [B, K, V]

            normalized_scores = torch.where(counts < total_seq_len,
                                            scores / (counts.float() + (kill_mask != 0.0).float()),
                                            scores / (counts.float() + 1.0) )  # [B, K, V]

            _, ind = torch.topk(normalized_scores.view(batch_size, -1), K)     # [B, K]

            sequence = torch.gather(tentative_seqs.view(batch_size, K * voc_dim, -1),
                                    1,
                                    ind.unsqueeze(-1).repeat(1, 1, tmp_len))      # [B, K, t + 2 + j]

            scores = torch.gather(scores.view(batch_size, K * voc_dim, -1),
                                  1,
                                  ind.unsqueeze(-1).repeat(1, 1, 1)).squeeze(-1)  # [B, K]

            # update mask
            mask = (sequence[:, : , -1] != EOS)  # [B, K]
            attn_mask = torch.cat([attn_mask, mask.unsqueeze(-1)], dim=-1)

            total_seq_len = total_seq_len + 1
            if torch.sum(mask) == 0:
                break

        return sequence

    def _batch_decoder(self, sequence, symbol_seq, length, K, decoding_method):

        num_samples = sequence.size(0)
        EOS = torch.full((num_samples, ),
                         self.data_loader.tokenizer.eos_token_id,
                         device=self.device)  # [B]

        for i in range(self.model.fix_len - length):

            attn_mask = torch.ones_like(sequence, device=self.device)
            if self.is_schema:
                logits = self.model.decoder(sequence, symbol_seq, attn_mask)  # [B, t, vocab_size]
            else:
                logits = self.model(sequence, attn_mask)
            logits = torch.nn.functional.log_softmax(logits, -1)
            batch_size, _, voc_dim = logits.shape

            if decoding_method == 'greedy':
                prediction = torch.argmax(logits[:, -1], dim=-1)  # shape [B]

            elif decoding_method == 'K-sampling':
                # get top K logits and indices
                top_logits, indices = torch.topk(logits[:, -1], k=K, sorted=False)  # shape [B, K]

                # sample from top K distribution
                distr = torch.distributions.Categorical(logits=top_logits)
                top_index = distr.sample()  # [B]
                prediction = torch.gather(indices, 1, top_index.unsqueeze(1)).squeeze()
            else:
                distr = torch.distributions.Categorical(logits=logits[:, -1])
                prediction = distr.sample()

            if i > 0:
                prediction = torch.where(sequence[:, -1] == EOS, EOS, prediction)

            sequence = torch.cat((sequence, prediction.unsqueeze(-1)), dim=-1)

            if torch.sum(prediction == EOS) == num_samples:
                break

        return sequence


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    path = './results'
    model_name = 'ptb/schema/2505_185447'
    version = 'best_model.pth'

    # load a model:
    schema = Model(path, model_name, device='cuda:0')

    schema.evaluate_metrics()
    schema.generate(decoding_method='K-beam_search')
    schema.interpolate()
    schema.print_reconstructions()
    schema.histogram()


    save_path_ = './results/'
    os.makedirs(save_path_, exist_ok=True)
    schema.save_rw_graph_label(save_path_, single_graph=False)
