import numpy as np


class ExponentialScheduler(object):

    def __init__(self, n_train_batches, n_epochs, **kwargs):
        self.max_steps = kwargs.get('max_steps', 1000)
        self.decay_rate = kwargs.get('decay_rate', 0.1)

    def __call__(self, step):
        return float(1. / (1. + np.exp(-self.decay_rate * (step - self.max_steps))))


class ExponentialIncrease(object):
    """
    Increases exponentially from zero to max_value
    """
    def __init__(self, n_train_batches, n_epochs, **kwargs):
        training_fraction_to_reach_max = kwargs.get('training_fraction_to_reach_max', 0.5)
        n_steps_to_rich_maximum = training_fraction_to_reach_max * n_train_batches * n_epochs

        self.max_value = kwargs.get('max_value', 1.0)
        self.decay_rate = -np.log(1. - 0.99) / n_steps_to_rich_maximum
        self.validation_value = kwargs.get('validation_value', 1.0)

    def __call__(self, step):
        return self.max_value * float(1. - np.exp(-self.decay_rate * step))


class ExponentialSchedulerGumbel(object):
    """
    Exponential annealing for Gumbel-Softmax temperature
    """
    def __init__(self, n_train_batches, n_epochs, **kwargs):
        self.temp_init = kwargs.get('temp_init')
        self.min_tau = kwargs.get('min_temp')

        training_fraction_to_reach_min = kwargs.get('training_fraction_to_reach_min', 0.5)
        n_steps_to_rich_minimum = training_fraction_to_reach_min * n_train_batches * n_epochs

        self.decay_rate = -np.log(self.min_tau) / n_steps_to_rich_minimum
        self.validation_value = kwargs.get('validation_value', 0.5)

    def __call__(self, step):
        t = np.maximum(self.temp_init * np.exp(-self.decay_rate * step), self.min_tau)
        return t


class ConstantScheduler(object):
    def __init__(self, n_train_batches, n_epochs, **kwargs):
        self.beta = kwargs.get('beta', 1.0)
        self.validation_value = kwargs.get('validation_value', 1.0)

    def __call__(self, step):
        return self.beta


class LinearScheduler(object):
    def __init__(self, n_train_batches, n_epochs, **kwargs):
        self.max_steps = kwargs.get('max_steps', 1000)
        self.start_value = kwargs.get('start_value', 0)
        print("start_value linear scheduler {}".format(self.start_value))

    def __call__(self, step):
        if self.start_value == 0:
            return min(1., float(step) / self.max_steps)
        else:
            return min(1., self.start_value + float(step) / self.max_steps * (1 - self.start_value))

class MultiplicativeScheduler(object):
    """
    Multiplies current value by multiplier each step until end_value is reached
    """

    def __init__(self, n_train_batches, n_epochs, **kwargs):
        self.start_value = kwargs.get('start_value', 1)
        self.end_value = kwargs.get('end_value', 0)
        self.multiplier = kwargs.get('multiplier', .9)

    def __call__(self, step):
        beta = self.start_value * self.multiplier**step
        return min(self.end_value, beta) if self.multiplier > 1 else max(self.end_value, beta)

class PeriodicScheduler(object):
    """
    """

    def __init__(self, n_train_batches, n_epochs, **kwargs):
        self.epoch_length = n_train_batches
        self.max_value = kwargs.get('max_value', 1.0)

        self.quarter_epoch_length = self.epoch_length * .25

    def __call__(self, step):
        step = step % self.epoch_length
        if step < self.epoch_length * .5:
            return 0
        elif step < self.epoch_length * .75:
            return (step - 2 * self.quarter_epoch_length) / self.quarter_epoch_length * self.max_value
        else:
            return self.max_value