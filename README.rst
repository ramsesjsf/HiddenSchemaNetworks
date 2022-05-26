====================
HiddenSchemaNetworks
====================


Installation
====
To install the code and all dependencies first install Python 3.8.
Then run in the project directory::

    pip install -e .

Experiments
====

Synthetic Data
____

**Training**

To reproduce any of the synthetic data experiments, run::

    python scripts/train_synthetic_schema.py -c experiments/synthetic/<config_file>

where ``<config_file>`` is one of the .yaml files in ``experiments/synthetic``.

**Evaluation**

To compute metrics for a whole directory of models, run::

    python scripts/evaluation_toolbox_synth.py -p results -n synth/<experiment_name>

where ``<experiment_name>`` is the name of the .yaml file, e.g. ``erdos_schema``


Real Data
____

**Training**

To train GPT2 or Schema models on the PTB, Yahoo, or Yelp data sets, run::

    python scripts/train_model.py -c experiments/<path>

with ``<path>`` path to a config file, e.g. ``ptb/gpt2.yaml``

For different numbers of symbols, you can modify the option *n_symbols* in the .yaml file,
and for different random walk lengths, change *rw_length* in the *encoder* section.

**Evaluation**

To compute metrics for a trained model saved in ``results/<path>``, run::

    python scripts/evaluation_toolbox.py -p results -n <path>

Functions for generating text, interpolations, and other graph statistics are all available in evaluation_toolbox.py

Note
====

This project has been set up using PyScaffold 3.2.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
