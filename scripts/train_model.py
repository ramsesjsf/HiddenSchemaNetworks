#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import click
import numpy as np
import torch
from hiddenschemanetworks.utils.helper import load_params, create_instance, expand_params, get_device
import hiddenschemanetworks

_logger = logging.getLogger(__name__)


@click.command()
@click.option('-c', '--config', 'cfg_path', required=True,
              type=click.Path(exists=True), help='path to config file')
@click.option('--quiet', 'log_level', flag_value=logging.WARNING, default=True)
@click.option('-v', '--verbose', 'log_level', flag_value=logging.INFO)
@click.option('-vv', '--very-verbose', 'log_level', flag_value=logging.DEBUG)
@click.option('-d', '--debug', 'debug', is_flag=True, default=False)
@click.option('-r', '--resume_training', 'resume', default=None,
              help='resume training from the last checkpoint')
@click.version_option(hiddenschemanetworks.__version__)

def main(cfg_path: Path, log_level: int, debug: bool, resume: bool):
    logging.basicConfig(stream=sys.stdout,
                        level=log_level,
                        datefmt='%Y-%m-%d %H:%M',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    params = load_params(cfg_path, _logger)
    gs_params = expand_params(params)
    train(debug, gs_params, params, resume)


def train(debug, gs_params, params, resume):
    num_workers = params['num_workers']
    if debug:
        train_in_debug(debug, gs_params, resume)
    else:
        train_parallel(num_workers, gs_params, resume)


def train_in_debug(debug, gs_params, resume):
    for search in gs_params:
        train_params(search, resume, debug)


def train_parallel(num_workers, gs_params, resume):
    p = Pool(num_workers)
    p.map(partial(train_params, resume=resume), gs_params)


def train_params(params, resume, debug=False):
    if debug:
        torch.manual_seed(int(params["seed"]))
        np.random.seed(int(params["seed"]))
    _logger.info("Name of the Experiment: " + params['name'])
    device = get_device(params)

    data_loader = create_instance('data_loader', params, device)

    if data_loader.vocab is not None:
        params['model']['args']['encoder']['args'].update({'vocab_size': len(data_loader.vocab)})
        if params['model']['args']['decoder']['args'] is None:
            params['model']['args']['decoder']['args'] = {'vocab_size': len(data_loader.vocab)}
        else:
            params['model']['args']['decoder']['args'].update({'vocab_size': len(data_loader.vocab)})

    model = create_instance('model', params, data_loader.pad_token_id, data_loader.fix_len)

    # Optimizers
    optimizers = init_optimizer(model, params)

    # Trainer
    trainer = create_instance('trainer', params, model, optimizers, False, resume, params, data_loader)
    best_model = trainer.train()
    with open(os.path.join(params['trainer']['logging']['logging_dir'], 'best_models.txt'), 'a+') as f:
        f.write(str(best_model) + "\n")


def init_optimizer(model, params):
    optimizers = dict()

    optimizer = create_instance('optimizer', params, model.parameters())
    optimizers['optimizer'] = {'opt': optimizer,
                               'grad_norm': params['optimizer'].get('gradient_norm_clipping', None),
                               'min_lr_rate': params['optimizer'].get('min_lr_rate', 1e-8)}

    return optimizers


if __name__ == '__main__':
    main()
