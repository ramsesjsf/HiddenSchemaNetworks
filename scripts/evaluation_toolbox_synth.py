from hiddenschemanetworks.utils.helper import create_instance, load_params
import logging
import torch
from collections import OrderedDict
from sklearn import metrics
import os
import numpy as np
from hiddenschemanetworks.models.languagemodels import SyntheticSchema
import click

@click.command()
@click.option('-p', '--path', 'path', required=True)
@click.option('-n', '--model_name', 'model_name', required=True)


def load_and_eval_models_in_dir(path, model_name):
    with torch.no_grad():
        device = 'cuda:0'
        nll_sum = torch.zeros(1).to(device)
        nll_sum_sq = torch.zeros(1).to(device)
        n_hits_sum = torch.zeros(1).to(device)
        n_hits_sum_sq = torch.zeros(1).to(device)
        n_edges_sum = torch.zeros(1).to(device)
        n_edges_sum_sq = torch.zeros(1).to(device)
        kl_graph_sum = torch.zeros(1).to(device)
        kl_graph_sum_sq = torch.zeros(1).to(device)
        kl_rws_sum = torch.zeros(1).to(device)
        kl_rws_sum_sq = torch.zeros(1).to(device)

        auc_sum = torch.zeros(1).to(device)
        auc_sum_sq = torch.zeros(1).to(device)

        count = 0
        for filename in os.listdir(os.path.join(path, model_name)):
            for subfilename in os.listdir(os.path.join(path, model_name, filename)):
                if subfilename == 'best_model.pth':
                    print("Model: " + model_name + os.sep + filename)
                    schema = Model(path, os.path.join(model_name, filename), version='best_model.pth', device='cuda:0')
                    nll, n_hits, n_edges, kl_graph, kl_rws = schema.evaluate_metrics(return_metrics=True)
                    nll_sum = nll_sum + nll
                    nll_sum_sq = nll_sum_sq + nll * nll
                    n_hits_sum = n_hits_sum + n_hits
                    n_hits_sum_sq = n_hits_sum_sq + n_hits * n_hits
                    n_edges_sum = n_edges_sum + 0.5 * n_edges
                    n_edges_sum_sq = n_edges_sum_sq + 0.5 * n_edges * 0.5 * n_edges
                    kl_graph_sum = kl_graph_sum + kl_graph
                    kl_graph_sum_sq = kl_graph_sum_sq + kl_graph * kl_graph
                    kl_rws_sum = kl_rws_sum + kl_rws
                    kl_rws_sum_sq = kl_rws_sum_sq + kl_rws * kl_rws

                    if isinstance(schema.model, SyntheticSchema):
                        auc = schema.evaluate_auc()
                        print('AUC: ', auc)
                        print('----------------')
                        auc_sum = auc_sum + auc
                        auc_sum_sq = auc_sum_sq + auc * auc

                    count += 1

        mean_nll = nll_sum / float(count)
        std_nll = torch.sqrt(torch.abs((nll_sum_sq - (nll_sum * nll_sum) / float(count)))/(float(count-1)))
        print("NLL =", mean_nll, "+/-", std_nll)

        mean_n_hits = n_hits_sum / float(count)
        std_n_hits = torch.sqrt((n_hits_sum_sq - (n_hits_sum * n_hits_sum) / float(count)) / (float(count - 1)))
        print("n hits =", mean_n_hits, "+/-", std_n_hits)

        mean_n_edges = n_edges_sum / float(count)
        std_n_edges = torch.sqrt((n_edges_sum_sq - (n_edges_sum * n_edges_sum) / float(count)) / (float(count - 1)))
        print("n_edges =", mean_n_edges, "+/-", std_n_edges)

        mean_kl_graph = kl_graph_sum / float(count)
        std_kl_graph = torch.sqrt((kl_graph_sum_sq - (kl_graph_sum * kl_graph_sum) / float(count)) / (float(count - 1)))
        print("kl_graph =", mean_kl_graph, "+/-", std_kl_graph)

        mean_kl_rws = kl_rws_sum / float(count)
        std_kl_rws = torch.sqrt((kl_rws_sum_sq - (kl_rws_sum * kl_rws_sum) / float(count)) / (float(count - 1)))
        print("kl_rws =", mean_kl_rws, "+/-", std_kl_rws)

        mean_auc = auc_sum / float(count)
        std_auc = torch.sqrt((auc_sum_sq - (auc_sum * auc_sum) / float(count)) / (float(count - 1)))
        print("auc =", mean_auc, "+/-", std_auc)

        print("count =", count)

class Model():

    def __init__(self, path, model_name, version='best_model.pth', device='cuda:0', data_loader=None):
        logger = logging.getLogger()
        self.model_name = model_name
        self.params = load_params(os.path.join(path, model_name, 'config.yaml'), logger)
        self.params['gpus'] = [device[-1]]

        self.device = torch.device(device)
        if data_loader is None:
            self.data_loader = create_instance('data_loader', self.params, self.device)
        else:
            self.data_loader = data_loader

        self.vocab = self.data_loader.vocab

        self.model = create_instance('model', self.params, self.data_loader).to(self.device)

        state_dict = torch.load(os.path.join(path, model_name, version))['model_state']

        try:
            self.model.load_state_dict(state_dict)
        except:
            pass
        self.model.eval()

    def evaluate_metrics(self, return_metrics=False):
        is_schema_model = isinstance(self.model, SyntheticSchema)
        with torch.no_grad():
            ppl = torch.zeros(1).to(self.device)
            nll_loss = torch.zeros(1).to(self.device)
            n_hits = torch.zeros(1).to(self.device)
            kl_graph = torch.zeros(1).to(self.device)
            kl_rws = torch.zeros(1).to(self.device)
            n_edges = torch.zeros(1).to(self.device)
            for i, minibatch in enumerate(self.data_loader.test):
                stats = self.model.validate_step(minibatch)
                nll_loss = nll_loss + stats['NLL-Loss']
                n_hits = n_hits + stats['number_of_hits']
                if is_schema_model:
                    n_edges = n_edges + torch.sum(self.sample_graph())
                    kl_graph = kl_graph + stats['KL-Graph']
                    kl_rws = kl_rws + stats['KL-RWs']
                    ppl = ppl + stats['PPL']

            print('----------------')
            print('NLL-Loss: ', nll_loss / float(len(self.data_loader.test)))
            print('number_of_hits: ', n_hits / float(len(self.data_loader.test)))
            print('number_of_edges: ', n_edges / float(len(self.data_loader.test)))
            print('KL-Graph: ', kl_graph / float(len(self.data_loader.test)))
            print('KL-RWs: ', kl_rws / float(len(self.data_loader.test)))
            print('PPL: ', ppl / float(len(self.data_loader.test)))
            print('----------------')

            if return_metrics:
                nll = nll_loss / float(len(self.data_loader.test))
                n_hits = n_hits / float(len(self.data_loader.test))
                n_edges = n_edges / float(len(self.data_loader.test))
                kl_graph = kl_graph / float(len(self.data_loader.test))
                kl_rws = kl_rws / float(len(self.data_loader.test))

                return nll, n_hits, n_edges, kl_graph, kl_rws

    def sample_graph(self, posterior_graph=True):
        with torch.no_grad():
            tau_graph = torch.tensor(0.5, device=self.device)
            if posterior_graph:
                adj_matrix = self.model.graph_generator(self.model.symbols, tau_graph, hard=True)[0]
            else:
                w = self.model.sample_gamma_var_prior()
                adj_matrix = self.model.sample_prior_graph()[0]
        return adj_matrix

    def evaluate_auc(self):
        with torch.no_grad():
            tau_graph = torch.tensor(0.5, device=self.device)
            link_prob = self.model.graph_generator(self.model.symbols,
                                                   tau_graph,
                                                   hard=True)[1]  # [n_symbols*(n_symbols-1)/2, 2]
            link_prob = link_prob[:, 0].cpu().numpy()
            ground_truth_graph = self.model.adj_matrix[self.model.graph_generator.mask_triu_elems].cpu().numpy()
            auc = metrics.roc_auc_score(ground_truth_graph, link_prob)
        return auc

    def saving_graph(self):
        with torch.no_grad():
            ground_truth_adj_matrix = self.model.adj_matrix.cpu().numpy()

            link_prob = self.model.graph_generator(self.model.symbols,
                                                   torch.tensor(0.5, device=self.model.device),
                                                   1,
                                                   hard=True)[1]
            link_prob = link_prob[:, 0]
            prob_matrix = torch.zeros(self.model.n_symbols, self.model.n_symbols, device=self.model.device)
            prob_matrix[self.model.triu_indices[0], self.model.triu_indices[1]] = link_prob
            prob_matrix = prob_matrix + torch.transpose(prob_matrix, 0, 1)
            prob_matrix = prob_matrix.cpu().numpy()

            ground_truth_symbols = self.model.ground_truth_word_prob.cpu().numpy()
            symbols = self.model.symbols.cpu().numpy()

            with open('./results/saved_graphs/ground_truth_adj_matrix.npy', 'wb') as f:
                np.save(f, ground_truth_adj_matrix)

            with open('./results/saved_graphs/link_probabilities.npy', 'wb') as f:
                np.save(f, prob_matrix)

            with open('./results/saved_graphs/ground_truth_symbols.npy', 'wb') as f:
                np.save(f, ground_truth_symbols)

            with open('./results/saved_graphs/learned_symbols.npy', 'wb') as f:
                np.save(f, symbols)

if __name__ == "__main__":
    #path = './results'
    #model_name = 'synth/schema'
    load_and_eval_models_in_dir()










