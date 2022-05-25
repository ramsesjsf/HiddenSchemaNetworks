#!/usr/bin/env python
# -*- coding: utf-8 -*-
import itertools
import logging
import os
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import click
import numpy as np
import torch
from IPython.core import ultratb
from tyche.utils.helper import load_params, create_instance, expand_params, get_device, get_independent_opts
import gentext

# fallback to debugger on error
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=1)

_logger = logging.getLogger(__name__)


@click.command()
@click.option('-c', '--config', 'cfg_path', required=True,
              type=click.Path(exists=True), help='path to config file')
@click.option('--quiet', 'log_level', flag_value=logging.WARNING, default=True)
@click.option('-v', '--verbose', 'log_level', flag_value=logging.INFO)
@click.option('-vv', '--very-verbose', 'log_level', flag_value=logging.DEBUG)
@click.option('-d', '--debug', 'debug', is_flag=True, default=False)
@click.option('-r', '--resume_training', 'resume', is_flag=True, default=False,
              help='resume training from the last checkpoint')

@click.version_option(gentext.__version__)
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
    model = create_instance('model', params, data_loader)

    # Optimizers
    optimizers = init_optimizer(model, params)

    # Trainer
    trainer = create_instance('trainer', params, model, optimizers, False, resume, params, data_loader)
    best_model = trainer.train()
    with open(os.path.join(params['trainer']['logging']['logging_dir'], 'best_models.txt'), 'a+') as f:
        f.write(str(best_model) + "\n")

def init_optimizer(model, params):
    optimizers = dict()

    independent_opt = get_independent_opts(params)
    if independent_opt:
        local_param = itertools.chain(*[model.encoder.parameters(), model.decoder.parameters()]) \
            if model.decoder is not None else model.encoder.parameters()

        optimizer = create_instance('optimizer', params, local_param)
        optimizers['optimizer'] = {'opt': optimizer,
                                   'grad_norm': params['optimizer'].get('gradient_norm_clipping', None),
                                   'min_lr_rate': params['optimizer'].get('min_lr_rate', 1e-8)}

        global_param = model.graph_generator.parameters()
        optimizer_global = create_instance('optimizer_global', params, global_param)
        optimizer_global.add_param_group({'params': model.symbols})

        optimizers['optimizer_global'] = {'opt': optimizer_global,
                                          'grad_norm': params['optimizer_global'].get('gradient_norm_clipping', None),
                                          'min_lr_rate': params['optimizer_global'].get('min_lr_rate', 1e-8)}
    else:
        optimizer = create_instance('optimizer', params, model.parameters())
        optimizers['optimizer'] = {'opt': optimizer,
                               'grad_norm': params['optimizer'].get('gradient_norm_clipping', None),
                               'min_lr_rate': params['optimizer'].get('min_lr_rate', 1e-8)}

    return optimizers


if __name__ == '__main__':
    main()
