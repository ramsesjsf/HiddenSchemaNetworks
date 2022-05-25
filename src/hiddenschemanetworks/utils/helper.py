# coding: utf-8

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import copy
import itertools
from functools import reduce
from importlib import import_module
from logging import Logger
from typing import List

import numpy as np
import torch as to
import torch as torch
import yaml
from scipy import linalg as la
import json


def create_class_instance(module_name, class_name, kwargs, *args):
    """Create an instance of a given class.

    :param module_name: where the class is located
    :param class_name:
    :param kwargs: arguments needed for the class constructor
    :returns: instance of 'class_name'

    """
    module = import_module(module_name)
    clazz = getattr(module, class_name)
    if kwargs is None:
        instance = clazz(*args)
    else:
        instance = clazz(*args, **kwargs)

    return instance


def create_instance(name, params, *args):
    """Creates an instance of class given configuration.

    :param name: of the module we want to create
    :param params: dictionary containing information how to instantiate the class
    :returns: instance of a class
    :rtype:

    """
    i_params = params[name]
    if type(i_params) is list:
        instance = [create_class_instance(
                p['module'], p['name'], p['args'], *args) for p in i_params]
    else:
        instance = create_class_instance(
                i_params['module'], i_params['name'], i_params['args'], *args)
    return instance


def create_nonlinearity(name):
    """
    Returns instance of non-linearity class (from torch.nn)
    """
    module = import_module("torch.nn")
    clazz = getattr(module, name)
    instance = clazz()

    return instance


def get_class_nonlinearity(name):
    """
    Returns non-linearity class (from torch.nn)
    """
    module = import_module("torch.nn")
    clazz = getattr(module, name)

    return clazz


def create_cost_function(name, *args):
    """
    Returns instance of cost functions (from torch.nn)
    """
    module = import_module("torch.nn")
    clazz = getattr(module, name)
    instance = clazz(*args)

    return instance


def load_params(path: str, logger: Logger) -> dict:
    """Loads experiment parameters from json file.

    :param path: to the json file
    :param logger:
    :returns: param needed for the experiment
    :rtype: dictionary

    """
    try:
        with open(path, "rb") as f:
            params = yaml.full_load(f)
        return params
    except Exception as e:
        logger.error(e)


def save_dict_to_file(obj: dict, path: str):
    """Save dictionary to a file

    :param obj: dict that is stored,
    :param path: to the location where the dictionary is stored.

    """
    with open(path, 'w') as f:
        json.dump(obj, f)


def to_one_hot(labels, num_classes):
    """
    Convert tensor of labels to one hot encoding of the labels.
    :param labels: to be encoded
    :param num_classes:
    :return:
    """
    shape = labels.size()
    shape = shape + (num_classes,)
    one_hot = torch.zeros(shape, dtype=torch.float, device=labels.device)
    dim = 1 if len(shape) == 2 else 2
    one_hot.scatter_(dim, labels.unsqueeze(-1), 1)
    return one_hot


def convert_tuples_2_list(arg):
    for key, value in arg.items():
        if isinstance(value, dict):
            convert_tuples_2_list(value)
        else:
            if isinstance(value, tuple):
                arg[key] = list(value)

    return arg


def unpack_cv_parameters(params, prefix=None):
    cv_params = []
    for key, value in params.items():
        if isinstance(value, dict):
            if prefix is None:
                prefix = key
            else:
                prefix = ".".join([prefix, key])
            param_pool = unpack_cv_parameters(value, prefix)
            if '.' in prefix:
                prefix = prefix.rsplit('.', 1)[0]
            else:
                prefix = None

            if len(param_pool) > 0:
                cv_params.extend(param_pool)
        elif isinstance(value, tuple) and len(value) != 0 and isinstance(value[0], dict):
            for ix, v in enumerate(value):
                if isinstance(v, dict):
                    if prefix is None:
                        prefix = key
                    else:
                        prefix = ".".join([prefix, key + f"#{ix}"])
                    param_pool = unpack_cv_parameters(v, prefix)
                    if '.' in prefix:
                        prefix = prefix.rsplit('.', 1)[0]
                    else:
                        prefix = None
                    if len(param_pool) > 0:
                        cv_params.extend(param_pool)
        elif isinstance(value, list):
            if prefix is None:
                prefix = key
            else:
                key = ".".join([prefix, key])
            cv_params.append([(key, v) for v in value])
    return cv_params


def dict_set_nested(d, keys, value):
    node = d
    key_count = len(keys)
    key_idx = 0

    for key in keys:
        key_idx += 1

        if key_idx == key_count:
            node[key] = value
            return d
        else:
            if "#" in key:
                key, _id = key.split("#")
                if not key in node:
                    node[key] = dict()
                    node = node[key][int(_id)]
                else:
                    node = node[key][int(_id)]
            else:
                if not key in node:
                    node[key] = dict()
                    node = node[key]
                else:
                    node = node[key]


def expand_params(params):
    """
    Expand the hyperparamers for grid search

    :param params:
    :return:
    """
    cv_params = []
    param_pool = unpack_cv_parameters(params)

    for i in list(itertools.product(*param_pool)):
        d = copy.deepcopy(params)
        name = d['name']
        for j in i:
            dict_set_nested(d, j[0].split("."), j[1])
            name += "_" + j[0] + "_" + str(j[1])
            d['name'] = name.replace('.args.', "_")
        d = convert_tuples_2_list(d)
        cv_params.append(d)
    if not cv_params:
        return [params] * params['num_runs']

    gs_params = []
    for p in cv_params:
        gs_params += [p] * p['num_runs']
    return gs_params


def get_cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def get_device(params: dict, rank: int = 0, logger: Logger = None) -> torch.device:
    """

    :param params:
    :param logger:
    :return: returns the device
    """
    gpus = params.get("gpus", [])
    if len(gpus) > 0:
        if not torch.cuda.is_available():
            if logger is not None:
                logger.warning("No GPU's available. Using CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:" + str(gpus[rank]))
    else:
        device = torch.device("cpu")
    return device


def gauss_legender_points(N=30):
    """ Returns the quadratures_nodes anb weights of the Gaussian-Lenegendre quadrature """
    beta = np.array([(n + 1.0) / np.sqrt((2.0 * n + 1.0) * (2.0 * n + 3.0))
                     for n in range(N - 1)], dtype=np.float32)
    M = np.diag(beta, -1) + np.diag(beta, 1)
    nodes, V = la.eigh(M, overwrite_a=True, overwrite_b=True)
    weight = 2 * V[0, :] ** 2
    return nodes, weight


def quadratures(f, a=-1, b=1, n=30):
    """
    Performing Legendre-Gauss quadrature integral approximation.

    :param f:
    :param a:
    :param b:
    :param n:
    :return:
    """
    nodes, weights = gauss_legender_points(n)
    w = to.tensor(weights.reshape(1, 1, -1))
    nodes = to.tensor(nodes.reshape(1, 1, -1))

    scale = (b - a) / 2.

    x = scale * nodes + (b + a) / 2.
    y = w * f(x)
    y = to.sum(scale * y, dim=-1)
    return y.type(dtype=to.float)


def greedy_sample_categorical(prob, one_hot=True):
    """
    Sample greedily from categorical distribution
    prob (Class probabilities): [B, n_classes]
    returns greedy samples, as one hot vectors if one_hot
    """
    n_classes = prob.shape[-1]
    z = torch.argmax(prob, dim=-1)
    if one_hot:
        z = torch.nn.functional.one_hot(z, num_classes=n_classes).float()

    return z


def gumbel_sample(shape, device, epsilon=1e-20):
    """
    Sample Gumbel(0,1)
    """
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + epsilon) + epsilon)


def gumbel_softmax_sample(pi, tau, device, epsilon=1e-12):
    """
    Sample Gumbel-softmax
    """
    y = torch.log(pi + epsilon) + gumbel_sample(pi.size(), device)
    return torch.nn.functional.softmax(y / tau, dim=-1)


def gumbel_softmax(pi, tau, device, hard=True):
    """
    Gumbel-Softmax distribution.
    Implementation from https://github.com/ericjang/gumbel-softmax.
    pi: [B, ..., n_classes] class probs of categorical z
    tau: temperature
    Returns [B, ..., n_classes] as a one-hot vector
    """
    y = gumbel_softmax_sample(pi, tau, device)
    if hard:
        shape = y.size()
        _, ind = y.max(dim=-1)  # [B, ...]
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        return (y_hard - y).detach() + y
    else:
        return y


def gumbel_softmax_argmax(pi, tau, device):
    """
    Gumbel-Softmax distribution.
    pi: [B, ..., n_classes] class probs of categorical z
    tau: temperature
    Returns [B, ..., n_classes] as a one-hot vector
            [B, ..., 1] (argmax over classes)
    """
    y = gumbel_softmax_sample(pi, tau, device)
    shape = y.size()
    _, ind = y.max(dim=-1)  # [B, ...]
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_ = (y_hard - y).detach() + y
    list_indices = torch.arange(shape[-1]).view(1, 1, shape[-1]).float()
    indices = torch.sum(y_ * list_indices, dim=-1)
    return y_, indices


def is_primitive(v):
    """
    Checks if v is of primitive type.
    """
    return isinstance(v, (int, float, bool, str))


def free_params(module):
    if type(module) is not list:
        module = [module]
    for m in module:
        for p in m.parameters():
            p.requires_grad = True


def frozen_params(module):
    if type(module) is not list:
        module = [module]
    for m in module:
        for p in m.parameters():
            p.requires_grad = False


def sum_dictionaries(dicts: List):
    """
    Sums the values of the common keys in dictionary.

    Parameters
    ----------
    dicts (list) dictionaries containing numeric values

    Returns
    -------
    dictionary with summed values
    """

    def reducer(accumulator, element):
        for key, value in element.items():
            accumulator[key] = accumulator.get(key, 0) + value
        return accumulator

    return reduce(reducer, dicts, {})


def sample(dist, mode=None, unk_idx=None):
    """
    Auxiliary sampling method.
    """
    if mode in ['sample-no-unk', 'greedy-no-unk'] and unk_idx is None:
        raise ValueError('Unknown index for the <unk> token!')
    if mode == 'greedy':
        _, _sample = torch.topk(dist, 1, dim=-1)
    elif mode == 'sample':
        sample_prob = torch.nn.functional.softmax(dist, dim=-1).squeeze(1)
        _sample = torch.multinomial(sample_prob, num_samples=1)
    elif mode == 'sample-no-unk':
        # reduce chances for <unk>
        dist[:, :, unk_idx] = dist.min()
        sample_prob = torch.nn.functional.softmax(dist, dim=-1).squeeze(1)
        _sample = torch.multinomial(sample_prob, num_samples=1)
    elif mode == 'greedy-no-unk':
        # prevent <unk>
        dist[:, :, unk_idx] = dist.min()
        _, _sample = torch.topk(dist, 1, dim=-1)
    else:
        raise ValueError(f'Unknown sampling mode = {mode}')

    _sample = _sample.squeeze()

    return _sample


def get_file_line_number(file_path: str) -> int:
    with open(file_path, 'r', encoding='utf-8') as f:
        buf_size = 1024 ** 2
        read_f = f.read
        lines: int = 0
        buf = read_f(buf_size)
        while buf:
            lines += buf.count('\n')
            buf = read_f(buf_size)
        return lines


def get_independent_opts(params: dict) -> bool:
    independent_opts = params["trainer"]["args"]["independent_optimizers"]
    return independent_opts
