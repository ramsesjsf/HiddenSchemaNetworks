import torch
from torch.autograd import Function
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from itertools import product
from networkx.drawing.nx_agraph import to_agraph


def graph_from_matrix(adjacency_matrix, show_isolated_nodes=True, weighted=True, self_loops=False):
    # Size of nodes may be misleading since the plots do not show loops,
    # i.e. edges (a, a) for node a.
    if isinstance(adjacency_matrix, torch.Tensor):
        adjacency_matrix = adjacency_matrix.numpy()
    if not self_loops:
        np.fill_diagonal(adjacency_matrix, 0)
    max_edge_weight = 10
    min_edge_weight = 0.1
    max_node_weight = 100
    fig = plt.figure(1)
    rows, cols = np.where(adjacency_matrix != 0)  # now rows == cols
    edges = zip(rows.tolist(), cols.tolist())
    if weighted:
        try:
            edge_weights = adjacency_matrix.reshape(-1)
            edge_weights = max_edge_weight * edge_weights[edge_weights != 0] / max(edge_weights) + min_edge_weight
        except Exception as e:
            print(f'Exception {e}: setting weighted=False')
            weighted = False
    if show_isolated_nodes:
        if weighted:
            try:
                weights = [float(x) for x in np.sum(adjacency_matrix, axis=1)]
                max_weight = max(weights)
                weights = [x * max_node_weight / max_weight for x in weights]
            except Exception as e:
                print(f'Exception {e}: setting weighted=False')
                weighted = False
        gr = nx.empty_graph(adjacency_matrix.shape[0])
    else:
        if weighted:
            try:
                weights = [float(x) for x in np.sum(adjacency_matrix, axis=0) if x != 0]
                max_weight = max(weights)
                weights = [x * max_node_weight / max_weight for x in weights]
            except Exception as e:
                print(f'Exception {e}: setting weighted=False')
                weighted = False
        gr = nx.Graph()
    gr.add_edges_from(edges)
    pos = {label: np.array((pos1, pos2)) for label, (pos1, pos2) in zip(range(64), product(range(8), range(8)))}
    pos = nx.circular_layout(gr)
    if weighted:
        nx.draw_networkx_nodes(gr, pos=pos, node_size=weights)
        nx.draw_networkx_edges(gr, pos=pos, width=edge_weights)
    else:
        nx.draw_networkx_nodes(gr, pos=pos)
        nx.draw_networkx_edges(gr, pos=pos)

    return fig


class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                                    inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
                           '`VectorQuantization`. The function `VectorQuantization` '
                           'is not differentiable. Use `VectorQuantizationStraightThrough` '
                           'if you want a straight-through estimator of the gradient.')


class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook)  # [B, T]
        indices_flatten = indices.view(-1)  # [B*T*D]
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
                                           index=indices_flatten)
        codes = codes_flatten.view_as(inputs)  # [B,T,D]

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            # map the outputs of grad to the respective embedding vectors
            # grad_codebook: size of codebook, has all the summed grads
            # grad_output gives a list of grad vectors (dim like codebook),
            # but no indication of which code they originally belong to
            # use indices to to map the grads to the correct positions in codebook
            # indices.shape [B*T]
            # grad_output_flatten.shape [B*T, D]

            grad_output_flatten = (grad_output.contiguous()
                                   .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)


class SampledVectorQuantizationStraightThrough(Function):

    @staticmethod
    def forward(ctx, inputs, codebook, m):

        B, T, D = inputs.shape
        K = codebook.shape[0]  # ?
        e = codebook.view(1, K, D).repeat(B * T, 1, 1)  # [B*T, K, D]
        z = inputs.view(B * T, 1, D).repeat(1, K, 1)  # [B*T, K, D]
        if m is None:
            m = B * T
        probs = torch.norm(z - e, dim=2)  # [B*T, K]
        indices = torch.distributions.Multinomial(m, probs).sample()  # [B*T, K]
        # indices_sampled = SampleMultinomial(m, probs).sample()  # [B*T, m]
        # indices_sampled.sum(axis=1) = m [B*T]
        # for each encoded symbol we get a K-dim vector of sampled counts

        m = indices.sum(axis=1)[0]
        # indices /= m
        ctx.save_for_backward(indices, codebook)
        ctx.mark_non_differentiable(indices)

        codes_flatten = torch.matmul(indices, codebook) / m  # [B*T]

        codes = codes_flatten.view_as(inputs)  # [B,T,D]

        return (codes, indices)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None
        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()

        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors  # [B, K], [K, D]
            B, K = indices.shape
            m = indices.sum(axis=1)[0]
            D = codebook.size(1)
            indices = indices.view(B, 1, K).repeat(1, D, 1) / m
            # !! torch.matmul(indices, embedding) ?
            grad_output_flatten = (grad_output.contiguous()
                                   .view(B, D))
            # loop over entire batch. mutliply grads output with corresponding
            # indices to remap it to the codebook entries that it was created from
            grad_codebook = (grad_output_flatten.view(B, 1, D) * indices.permute(0, 2, 1)).sum(dim=0)

        return (grad_inputs, grad_codebook, None)


svq_st = SampledVectorQuantizationStraightThrough.apply
vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply
__all__ = [vq, vq_st]


def clip_grad_norm(parameters, optimizer: dict) -> None:
    if optimizer['grad_norm'] is not None:
        torch.nn.utils.clip_grad_norm_(parameters, optimizer['grad_norm'])
