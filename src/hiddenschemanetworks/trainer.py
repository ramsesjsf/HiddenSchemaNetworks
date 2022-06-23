import datetime
import json
import logging
import os
from abc import ABCMeta
from collections import ChainMap
from typing import Any, Dict

import matplotlib
import torch
import numpy as np
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from hiddenschemanetworks.utils.helper import is_primitive, create_instance, get_device


class BaseTrainingProcedure(metaclass=ABCMeta):

    def __init__(self, model: torch.nn.Module, optimizer: dict, distributed: bool, resume: bool, params: dict, data_loader, train_logger=None,
                 **kwargs):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.data_loader = data_loader
        self.optimizer: dict = optimizer
        self.params: dict = params
        self.rank: int = 0
        self.world_size: int = -1
        self.device = get_device(params, self.rank, self.logger)
        self.model = model.to(self.device)

        self._prepare_dirs()
        self._save_params()
        self.t_logger = self._setup_logging()
        self.summary = SummaryWriter(self.tensorboard_dir)

        self.start_epoch: int = 0
        self.n_epochs: int = self.params['trainer']['epochs']
        self.save_after_epoch: int = self.params['trainer']['args']['save_after_epoch']
        self.batch_size: int = self.params['data_loader']['args']['batch_size']
        self.bm_metric: str = self.params['trainer']['args']['bm_metric']

        self.lr_schedulers = self.__init_lr_schedulers()

        if 'schedulers' in self.params['trainer']['args']:
            self.schedulers = dict()
            schedulers_ = create_instance('schedulers',
                                          self.params['trainer']['args'],
                                          *(data_loader.n_train_batches,
                                            self.n_epochs,
                                            ))
            if type(schedulers_) is not list:
                schedulers_ = [schedulers_]
            for a, b in zip(self.params['trainer']['args']['schedulers'], schedulers_):
                self.schedulers[a["label"]] = b
        else:
            self.schedulers = None

        self.data_loader = data_loader
        self.n_train_batches: int = data_loader.n_train_batches
        self.n_validate_batches: int = data_loader.n_validate_batches
        self.n_test_batches: int = data_loader.n_test_batches

        self.global_step: int = 0
        self.best_model = {'train_loss': float('inf'),
                           'val_loss': float('inf'),
                           'train_metric': float('inf'),
                           'val_metric': float('inf')}

        self.train_logger = train_logger
        if resume:
            self._resume_check_point(resume)

    def __init_lr_schedulers(self):
        lr_schedulers = self.params['trainer']['args'].get('lr_schedulers', None)
        if lr_schedulers is None:
            return None
        schedulers = dict()
        lr_schedulers = ChainMap(*lr_schedulers)
        for opt_name, scheduler in lr_schedulers.items():
            opt_scheduler = create_instance(opt_name, lr_schedulers, self.optimizer[opt_name]['opt'])
            schedulers[opt_name] = {'counter': scheduler.get('counter'),
                                    'default_counter': scheduler.get('counter'),
                                    'scheduler': opt_scheduler}
        return schedulers

    def train(self):
        e_bar = tqdm(
                desc=f'Rank {self.rank}, Epoch: ',
                total=self.n_epochs,
                unit='epoch',
                initial=self.start_epoch,
                position=self.rank * 2,
                ascii=True,
                leave=True)

        for epoch in range(self.start_epoch, self.n_epochs):
            train_log = self._train_epoch(epoch)
            validate_log = self._validate_epoch(epoch)
            test_log = self._test_epoch(epoch)
            self._anneal_lr(validate_log)
            self._update_p_bar(e_bar, train_log, validate_log, test_log)
            self._booking_model(epoch, train_log, validate_log)
            if self._check_early_stopping(validate_log):
                break
        self._clear_logging_resources(e_bar)
        return self.best_model

    def _clear_logging_resources(self, e_bar: tqdm) -> None:
        self.summary.flush()
        self.summary.close()
        e_bar.close()

    def _booking_model(self, epoch: int, train_log: dict, validate_log: dict) -> None:
        self._check_and_save_best_model(train_log, validate_log)
        if epoch % self.save_after_epoch == 0 and epoch != 0:
            self._save_check_point(epoch)

    def _anneal_lr(self, validate_log: dict) -> None:
        if self.lr_schedulers is not None:
            if validate_log[self.bm_metric] > self.best_model['val_metric']:
                for key, value in self.lr_schedulers.items():
                    if value['counter'] > 0:
                        value['counter'] -= 1
                    else:
                        value['scheduler'].step()
                        value['counter'] = value['default_counter']
            else:
                for key, value in self.lr_schedulers.items():
                    value['counter'] = value['default_counter']

    def _check_early_stopping(self, log: dict) -> bool:
        cond = list(filter(lambda x: x['opt'].param_groups[0]["lr"] < float(x['min_lr_rate']), self.optimizer.values()))
        loss = log['loss']
        return len(cond) != 0 or np.isinf(loss) or np.isnan(loss)

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        p_bar = tqdm(
                desc=f'Rank {self.rank}, Training batch: ',
                total=self.n_train_batches,
                unit='batch',
                leave=False,
                ascii=True,
                position=self.rank * 2 + 1
        )
        epoch_stats = None
        for batch_idx, data in enumerate(self.data_loader.train):
            batch_stats = self._train_step(data, batch_idx, epoch, p_bar)
            epoch_stats = self._update_stats(epoch_stats, batch_stats)
        p_bar.close()
        del p_bar
        epoch_stats = self._normalize_stats(self.n_train_batches, epoch_stats)
        self._log_epoch('train/epoch/', epoch_stats)

        return epoch_stats

    def _train_step(self, minibatch: Any, batch_idx: int, epoch: int, p_bar: tqdm) -> dict:
        stats = self.model.train_step(minibatch, self.optimizer, self.global_step, scheduler=self.schedulers)
        self._update_step_p_bar(p_bar, stats)
        stats = self.tensor_2_item(stats)

        self._log_step('train', epoch, batch_idx, self.data_loader.train_set_size, stats)
        self.global_step += 1

        return stats

    def _validate_epoch(self, epoch: int) -> dict:
        self.model.eval()
        with torch.no_grad():
            p_bar = tqdm(
                    desc=f"Rank {self.rank}, Validation batch: ",
                    total=self.n_validate_batches,
                    unit="batch",
                    leave=False,
                    ascii=True,
                    position=self.rank * 2 + 1)

            epoch_stats = None
            for batch_idx, data in enumerate(self.data_loader.validate):
                batch_stats = self._validate_step(data, batch_idx, epoch, p_bar)
                epoch_stats = self._update_stats(epoch_stats, batch_stats)
            p_bar.close()
            del p_bar
            epoch_stats = self._normalize_stats(self.n_validate_batches, epoch_stats)
            self._log_epoch('validate/epoch/', epoch_stats)

            return epoch_stats

    def _validate_step(self, minibatch: dict, batch_idx: int, epoch: int, p_bar: tqdm) -> dict:
        stats = self.model.validate_step(minibatch)
        self._update_step_p_bar(p_bar, stats)
        stats = self.tensor_2_item(stats)

        self._log_step('validate', epoch, batch_idx, self.data_loader.validation_set_size, stats)

        return stats

    def _test_epoch(self, epoch: int) -> dict:
        self.model.eval()
        with torch.no_grad():
            p_bar = tqdm(
                    desc=f'Rank {self.rank}, Test batch: ',
                    total=self.n_test_batches,
                    unit='batch',
                    ascii=True,
                    position=self.rank * 2 + 1,
                    leave=False)

            epoch_stats = None
            for batch_idx, data in enumerate(self.data_loader.test):
                batch_stat = self._test_step(data, batch_idx, epoch, p_bar)
                epoch_stats = self._update_stats(epoch_stats, batch_stat)
            p_bar.close()
            del p_bar

            self._normalize_stats(self.n_test_batches, epoch_stats)
            self._log_epoch('test/epoch/', epoch_stats)

        return epoch_stats

    def _test_step(self, minibatch: dict, batch_idx: int, epoch: int, p_bar: tqdm) -> dict:
        stats = self.model.validate_step(minibatch)
        self._update_step_p_bar(p_bar, stats)
        stats = self.tensor_2_item(stats)
        self._log_step('test', epoch, batch_idx, self.data_loader.test_set_size, stats)

        return stats

    @staticmethod
    def _update_stats(epoch_stat: dict, batch_stat: dict) -> dict:
        if epoch_stat is None:
            return batch_stat.copy()
        for k, v in batch_stat.items():
            epoch_stat[k] += v

        return epoch_stat

    @staticmethod
    def _normalize_stats(n_batches: int, statistics: dict) -> dict:
        for k, v in statistics.items():
            if is_primitive(v):
                statistics[k] /= n_batches
        return statistics

    def _log_epoch(self, log_label: str, statistics: dict) -> None:
        for k, v in statistics.items():
            if is_primitive(v):
                self.summary.add_scalar(log_label + k, v, self.global_step)
            elif isinstance(v, list) and isinstance(v[0], int):
                self.summary.add_histogram(log_label + k, v, self.global_step)
            elif isinstance(v, matplotlib.figure.Figure):
                self.summary.add_figure(log_label + k, figure=v, global_step=self.global_step)

    def _prepare_dirs(self) -> None:
        trainer_par = self.params['trainer']
        start_time = datetime.datetime.now().strftime('%d%m_%H%M%S')
        name = self.params['name']
        if len(name) > 200:
            name = "_".join([i if i.isdigit() else i[0:3] for i in name.split('_')])
        self.checkpoint_dir = os.path.join(trainer_par['save_dir'], name, start_time)
        self.logging_dir = os.path.join(trainer_par['logging']['logging_dir'], name, start_time)
        self.tensorboard_dir = os.path.join(trainer_par['logging']['tensorboard_dir'], name, start_time)

        os.makedirs(self.logging_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)

    def _save_params(self):
        params_path = os.path.join(self.logging_dir, 'config.yaml')
        self.logger.info(f'saving config into {params_path}')
        yaml.dump(self.params, open(params_path, 'w'), default_flow_style=False)

    def _save_model(self, file_name: str, **kwargs) -> None:
        model_type = type(self.model).__name__
        state = {
            'model_type': model_type,
            'epoch': kwargs.get('epoch'),
            'model_state': self.model.state_dict(),
            'params': self.params,
            'vocab': pickle.dumps(self.data_loader.vocab)
        }
        for key in self.optimizer:
            state[key] = self.optimizer[key]['opt'].state_dict()

        torch.save(state, file_name)

    def _save_model_parameters(self, file_name):
        with open(file_name, 'w') as f:
            json.dump(self.params, f, indent=4)

    def _save_check_point(self, epoch: int) -> None:
        file_name = os.path.join(self.checkpoint_dir,
                                 'checkpoint-epoch{}.pth'.format(epoch))
        self.t_logger.info('Saving checkpoint: {} ...'.format(file_name))
        self._save_model(file_name, epoch=epoch)

    def _save_best_model(self) -> None:
        file_name = os.path.join(self.checkpoint_dir,
                                 'best_model.pth')
        self.t_logger.info('Saving best model ...')
        self._save_model(file_name)

    def _resume_check_point(self, path: str) -> None:
        self.logger.info('Loading checkpoint: {} ...'.format(path))
        if torch.cuda.is_available() is False:
            state = torch.load(path, map_location='cpu')
        else:
            state = torch.load(path)
        self.params = state['params']
        if state['epoch'] is None:
            self.start_epoch = 1
        else:
            self.start_epoch = state['epoch'] + 1
        self.model.load_state_dict(state['model_state'])
        for key in self.optimizer:
            self.optimizer[key].load_state_dict(state[key])
        self.logger.info('Finished loading checkpoint: {} ...'.format(path))

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger('train_logger')
        logger.propagate = False
        logger.setLevel(logging.INFO)

        file_name = os.path.join(self.logging_dir, 'train.log')
        fh = logging.FileHandler(file_name)
        formatter = logging.Formatter(
                self.params['trainer']['logging']['formatters']['simple'])
        fh.setLevel(logging.INFO)

        fh.setFormatter(formatter)
        logger.addHandler(fh)

        return logger

    def _log_step(self, step_type: str, epoch: int, batch_idx: int, data_len: int, stats: dict) -> None:
        log = self._build_raw_log_str(f'{step_type} epoch', batch_idx, epoch, stats, float(data_len), self.batch_size)
        self.t_logger.info(log)
        for k, v in stats.items():
            if is_primitive(v):
                self.summary.add_scalar(f'{step_type}/batch/' + k, v, self.global_step)

    @staticmethod
    def _build_raw_log_str(prefix: str, batch_idx: int, epoch: int, logs: dict, data_len: float, batch_size: int):
        sb = prefix + ': {} [{}/{} ({:.0%})]'.format(
                epoch,
                batch_idx * batch_size,
                data_len,
                100.0 * batch_idx / data_len)
        for k, v in logs.items():
            if is_primitive(v):
                sb += ' {}: {:.6f}'.format(k, v)
        return sb

    def _check_and_save_best_model(self, train_log: dict, validate_log: dict) -> None:
        if validate_log[self.bm_metric] < self.best_model['val_metric']:
            self._save_best_model()
            self._update_best_model_flag(train_log, validate_log)

    def _update_p_bar(self, e_bar: tqdm, train_log: dict, validate_log: dict, test_log: dict) -> None:
        e_bar.set_postfix_str(
                f"train loss: {train_log['loss']:4.4g} train {self.bm_metric}: {train_log[self.bm_metric]:4.4g}, "
                f"validation loss: {validate_log['loss']:4.4g}, validation {self.bm_metric}: {validate_log[self.bm_metric]:4.4g} "
                f"test loss: {test_log['loss']:4.4g}, test {self.bm_metric}: {test_log[self.bm_metric]:4.4g}")
        e_bar.update()

    @staticmethod
    def _update_step_p_bar(p_bar: tqdm, stats: dict):
        log_str = ''
        for key, value in stats.items():
            if isinstance(value, tuple) or (type(value) is torch.Tensor and len(value.size()) >= 1):
                continue
            log_str += f"{key}: {value.item():4.4g} "

        p_bar.update()
        p_bar.set_postfix_str(log_str)

    def _update_best_model_flag(self, train_log: dict, validate_log: dict) -> None:
        self.best_model['train_loss'] = train_log['loss']
        self.best_model['val_loss'] = validate_log['loss']
        self.best_model['train_metric'] = train_log[self.bm_metric]
        self.best_model['val_metric'] = validate_log[self.bm_metric]
        self.best_model['name'] = self.params['name']

    @staticmethod
    def tensor_2_item(stats):
        for key, value in stats.items():
            if type(value) is torch.Tensor and len(value.size()) == 0:
                stats[key] = value.item()
        return stats

class TextTrainer(BaseTrainingProcedure):
    def __init__(self, model, optimizer, resume, params, data_loader, train_logger=None, **kwargs):
        super(TextTrainer, self).__init__(model, optimizer, resume, params, data_loader, train_logger, **kwargs)
        self.reconstruction_every = kwargs.pop("reconstruction_every")
        self.num_of_rec_sentences = kwargs.pop("num_rec_sentences")
        self.num_of_samples = kwargs.pop("num_samples")
        self.num_of_interpolation_samples = kwargs.pop("num_interpolation_samples")
        self.num_interpolation_steps = kwargs.pop("num_interpolation_steps")

    def _train_step(self, minibatch: dict, batch_idx: int, epoch: int, p_bar) -> Dict:
        minibatch = {key: value.to(self.device) for key, value in minibatch.items()}
        stats = super()._train_step(minibatch, batch_idx, epoch, p_bar)
        self._log_reconstruction('train/', stats['reconstruction'][0], stats['reconstruction'][1], False)
        self._log_interpolation('', minibatch, False)
        del stats['reconstruction']

        return stats

    def _validate_step(self, minibatch: dict, batch_idx: int, epoch: int, p_bar) -> Dict:
        minibatch = {key: value.to(self.device) for key, value in minibatch.items()}
        stats = super()._validate_step(minibatch, batch_idx, epoch, p_bar)
        self._log_reconstruction('validate/', stats['reconstruction'][0], stats['reconstruction'][1], False)
        del stats['reconstruction']

        return stats

    def _test_step(self, minibatch: dict, batch_idx: int, epoch: int, p_bar) -> Dict:
        minibatch = {key: value.to(self.device) for key, value in minibatch.items()}
        stats = super()._validate_step(minibatch, batch_idx, epoch, p_bar)
        self._log_reconstruction('test/', stats['reconstruction'][0], stats['reconstruction'][1], False)
        del stats['reconstruction']

        return stats

    def _log_sample(self, tag, force_log=False):
        if self.global_step % self.reconstruction_every != 0 and not force_log:
            return None
        B = self.num_of_samples
        if B != 0:
            sample, _, _ = self.model.sample(B)

            transformer = self.__get_text_transformer()
            log = []
            for smpl in transformer(sample):
                log.append(smpl.replace('<unk>', '|unk|'))
            log = "\n\n ------------------------------------- \n\n".join(log)

            self.summary.add_text(tag + 'samples', log, self.global_step)

    def _log_interpolation(self, tag, sample, force_log=False):
        if self.global_step % self.reconstruction_every != 0 and not force_log:
            return None
        n = self.num_of_interpolation_samples
        if n != 0:
            seq_len, _ix = torch.sort(sample['length'], descending=True)
            input_seq = sample['input'][_ix]

            interpolations = self.model.interpolate(start=(input_seq[:n], seq_len[:n]),
                                                    end=(input_seq[1:n + 1], seq_len[1:n + 1]),
                                                    num_steps=self.num_interpolation_steps)

            log = []
            transformer = self.__get_text_transformer()
            for i in range(n):
                log_sample = []
                smpl = transformer(interpolations[i])
                smpl[0] = f'<strong>{smpl[0]}</strong>'
                smpl[-1] = f'<strong>{smpl[-1]}</strong>'
                log_sample.extend(list(map(lambda x: x.replace('<unk>', '|unk|'), smpl)))
                log_sample = "\n\n".join(log_sample)
                log.append(log_sample)
            log = "\n\n ------------------------------------------------------------- \n\n".join(log)
            self.summary.add_text(tag + 'interpolation', log, self.global_step)

    def __get_text_transformer(self):
        if not hasattr(self.data_loader.train.dataset, 'fields'):
            transformer = self.data_loader.train.dataset.reverse
        else:
            transformer = self.data_loader.train.dataset.fields['text']
        return transformer

    def _log_reconstruction(self, tag, prediction, target, force_log=False):
        if self.global_step % self.reconstruction_every != 0 and not force_log:
            return None
        B = prediction.size(0)
        sample_size = min(B, self.num_of_rec_sentences)
        ix_ = np.random.randint(0, B, sample_size)
        transformer = self.__get_text_transformer()

        t = transformer(target[ix_])
        r = transformer(prediction[ix_])

        log = []
        for i, j in zip(t, r):
            log_string = "Org: " + "\n\n Rec: ".join([i, j])
            log.append(log_string.replace('<unk>', '|unk|'))
        log = "\n\n ------------------------------------- \n\n".join(log)

        self.summary.add_text(tag + 'reconstruction', log, self.global_step)

BaseTrainingProcedure.register(TextTrainer)

class TrainerSimpleSchema(TextTrainer):
    def __init__(self, model, optimizer, resume, params, data_loader, train_logger=None, **kwargs):
        super(TrainerSimpleSchema, self).__init__(model, optimizer, resume, params, data_loader,
                                                  train_logger=train_logger, **kwargs)

        self.heat_map = True

    def _train_step(self, minibatch: Any, batch_idx: int, epoch: int, p_bar) -> Dict:


        stats = self.model.train_step(minibatch, self.optimizer, self.global_step, scheduler=self.schedulers)

        self._update_step_p_bar(p_bar, stats)
        stats = self.tensor_2_item(stats)

        self._log_step('train', epoch, batch_idx, self.data_loader.train_set_size, stats)
        self.global_step += 1

        if len(stats['reconstruction']) > 2:
            self._log_reconstruction_dummy('train/', stats['reconstruction'][0], stats['reconstruction'][1],
                                           stats['reconstruction'][2], stats['reconstruction'][3],
                                           stats['reconstruction'][4], False)
        else:
            self._log_reconstruction_dummy_2('train/', stats['reconstruction'][0], stats['reconstruction'][1], False)

        del stats['reconstruction']

        return stats

    def _validate_step(self, minibatch: dict, batch_idx: int, epoch: int, p_bar) -> Dict:
        stats = self.model.validate_step(minibatch)
        self._update_step_p_bar(p_bar, stats)
        stats = self.tensor_2_item(stats)
        self._log_step('validate', epoch, batch_idx, self.data_loader.validation_set_size, stats)
        if len(stats['reconstruction']) > 2:
            self._log_reconstruction_dummy('validate/', stats['reconstruction'][0], stats['reconstruction'][1],
                                           stats['reconstruction'][2], stats['reconstruction'][3],
                                           stats['reconstruction'][4], True)
        else:
            self._log_reconstruction_dummy_2('validate/', stats['reconstruction'][0], stats['reconstruction'][1], True)
        del stats['reconstruction']
        return stats

    def _test_step(self, minibatch: dict, batch_idx: int, epoch: int, p_bar) -> Dict:
        stats = self.model.validate_step(minibatch)
        self._update_step_p_bar(p_bar, stats)
        stats = self.tensor_2_item(stats)
        self._log_step('test', epoch, batch_idx, self.data_loader.test_set_size, stats)
        if len(stats['reconstruction']) > 2:
            self._log_reconstruction_dummy('test/', stats['reconstruction'][0], stats['reconstruction'][1],
                                           stats['reconstruction'][2], stats['reconstruction'][3],
                                           stats['reconstruction'][4], True)
        else:
            self._log_reconstruction_dummy_2('test/', stats['reconstruction'][0], stats['reconstruction'][1], True)

        del stats['reconstruction']

        if self.heat_map:
            self._log_heatmap_wordprob(batch_idx)

        return stats

    def _log_reconstruction_dummy(self, tag, prediction, target_a, target_b, z_, z, force_log=False):
        if self.global_step % self.reconstruction_every != 0 and not force_log:
            return None
        z = z.long().cpu().numpy().astype('str').tolist()
        z_ = z_.long().cpu().detach().numpy().astype('str').tolist()
        str_ = ''
        for i in range(self.num_of_rec_sentences):
            str_predic =  'output : '
            str_target_a = 'token A: '
            str_target_b = 'token B: '
            str_z_ = 'pred schema: '
            str_z =  'true schema: '
            for j in range(self.model.fix_len):
                str_predic += self.data_loader.vocab.itos[prediction[i, j]] + " "
                str_target_a += self.data_loader.vocab.itos[target_a[i, j]] + " "
                str_target_b += self.data_loader.vocab.itos[target_b[i, j]] + " "
                str_z_ += z_[i][j] + ", "
                str_z += z[i][j] + ", "
            str_ += str_predic + "\n\n " + str_target_a + "\n\n " + str_target_b + "\n\n " + \
                    str_z_ + "\n\n " + str_z + \
                    "\n\n ------------------------------------- \n\n"

        self.summary.add_text(tag + 'reconstruction', str_, self.global_step)

    def _log_reconstruction_dummy_2(self, tag, prediction, target, force_log=False):
        if self.global_step % self.reconstruction_every != 0 and not force_log:
            return None
        str_ = ''
        for i in range(self.num_of_rec_sentences):
            str_predic =  'prediction : '
            str_target = 'target : '
            for j in range(self.model.fix_len):
                str_predic += self.data_loader.vocab.itos[prediction[i, j]] + " "
                str_target += self.data_loader.vocab.itos[target[i, j]] + " "

            str_ += str_predic + "\n\n " + str_target + \
                    "\n\n ------------------------------------- \n\n"

        self.summary.add_text(tag + 'reconstruction', str_, self.global_step)

    def _log_heatmap_wordprob(self, batch_idx):
        if batch_idx == self.n_test_batches - 1:
            ground_truth_word_prob = self.model.ground_truth_word_prob.cpu().numpy()
            word_prob = torch.softmax(self.model.symbols, dim=-1).cpu().numpy()
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
            sns.heatmap(ground_truth_word_prob, ax=ax1)
            sns.heatmap(word_prob, ax=ax2)
            sns.heatmap(ground_truth_word_prob-word_prob, ax=ax3)
            self.summary.add_figure(tag='heatmaps', figure=fig, global_step=self.global_step)

BaseTrainingProcedure.register(TrainerSimpleSchema)

class TrainerRealSchema(BaseTrainingProcedure):
    def __init__(self, model, optimizer, distributed, resume, params, data_loader, train_logger=None, **kwargs):
        super(TrainerRealSchema, self).__init__(model, optimizer, distributed, resume, params, data_loader,
                                                  train_logger=train_logger, **kwargs)

        self.reconstruction_every = kwargs.pop("reconstruction_every")
        self.num_of_rec_sentences = kwargs.pop("num_rec_sentences")

    def train(self):
        e_bar = tqdm(
                desc=f'Rank {self.rank}, Epoch: ',
                total=self.n_epochs,
                unit='epoch',
                initial=self.start_epoch,
                position=self.rank * 2,
                ascii=True,
                leave=True)

        epoch_ = 0
        for epoch in range(self.start_epoch, self.n_epochs):
            epoch_ = epoch
            validate_log = self._do_epoch(epoch, e_bar)
            if self._check_early_stopping(validate_log):
                break
        self._clear_logging_resources(e_bar)
        return self.best_model

    def _do_epoch(self, epoch, e_bar):
        train_log = self._train_epoch(epoch)
        validate_log = self._validate_epoch(epoch)
        test_log = self._test_epoch(epoch)
        self._anneal_lr(validate_log)
        self._update_p_bar(e_bar, train_log, validate_log, test_log)
        self._booking_model(epoch, train_log, validate_log)
        return validate_log

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        p_bar = tqdm(
                desc=f'Rank {self.rank}, Training batch: ',
                total=self.n_train_batches,
                unit='batch',
                leave=False,
                ascii=True,
                position=self.rank * 2 + 1
        )
        epoch_stats = None

        for batch_idx, data in enumerate(self.data_loader.train):
            batch_stats = self._train_step(data, batch_idx, epoch, p_bar)
            epoch_stats = self._update_stats(epoch_stats, batch_stats)
        p_bar.close()
        del p_bar
        epoch_stats = self._normalize_stats(self.n_train_batches, epoch_stats)
        self._log_epoch('train/epoch/', epoch_stats)

        return epoch_stats

    def _train_step(self, minibatch: dict, batch_idx: int, epoch: int, p_bar) -> Dict:
        minibatch = {key: value.to(self.device) for key, value in minibatch.items()}
        stats = super()._train_step(minibatch, batch_idx, epoch, p_bar)

        self._log_symbols('train/', stats['symbols'], False)
        del stats['symbols']
        self._log_reconstruction('train/', stats['reconstruction'][0], stats['reconstruction'][1], False)
        del stats['reconstruction']

        return stats

    def _validate_step(self, minibatch: dict, batch_idx: int, epoch: int, p_bar) -> Dict:
        minibatch = {key: value.to(self.device) for key, value in minibatch.items()}
        stats = super()._validate_step(minibatch, batch_idx, epoch, p_bar)

        self._log_symbols('validate/', stats['symbols'], True)
        del stats['symbols']
        self._log_reconstruction('validate/', stats['reconstruction'][0], stats['reconstruction'][1], True)
        del stats['reconstruction']

        return stats

    def _test_step(self, minibatch: dict, batch_idx: int, epoch: int, p_bar) -> Dict:
        minibatch = {key: value.to(self.device) for key, value in minibatch.items()}
        stats = super()._test_step(minibatch, batch_idx, epoch, p_bar)

        self._log_symbols('test/', stats['symbols'], True)
        del stats['symbols']
        self._log_reconstruction('test/', stats['reconstruction'][0], stats['reconstruction'][1], True)
        del stats['reconstruction']

        return stats

    def _log_symbols(self, tag, symbols, force_log=False):
        if self.global_step % self.reconstruction_every != 0 and not force_log:
            return None
        z = symbols.long().cpu().numpy().astype('str').tolist()
        str_ = ''
        for i in range(self.num_of_rec_sentences):
            str_z = 'Schema: '
            for j in range(self.model.encoder.rw_length):
                str_z += z[i][j] + ", "
            str_ += str_z + "\n\n ------------------------------------- \n\n"

        self.summary.add_text(tag + 'symbols', str_, self.global_step)

    def _log_reconstruction(self, tag, prediction, target, force_log=False):
        if self.global_step % self.reconstruction_every != 0 and not force_log:
            return None
        B = prediction.size(0)
        sample_size = min(B, self.num_of_rec_sentences)
        ix_ = np.random.randint(0, B, sample_size)

        t = self.data_loader.train.dataset.reverse(target[ix_])
        r = self.data_loader.train.dataset.reverse(prediction[ix_])

        log = []
        for i, j in zip(t, r):
            log_string = "Org: " + "\n\n Rec: ".join([i, j])
            log.append(log_string.replace('<unk>', '|unk|'))
        log = "\n\n ------------------------------------- \n\n".join(log)

        self.summary.add_text(tag + 'reconstruction', log, self.global_step)

BaseTrainingProcedure.register(TrainerRealSchema)
