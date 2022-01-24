ARCA23K Baseline System
=======================

This is the source code for the baseline system associated with the
`ARCA23K`__ dataset. Details about ARCA23K and the baseline system can
be found in our `DCASE2021 paper`__ [1]_. 

__ https://zenodo.org/record/5117901
__ https://arxiv.org/abs/2109.09227


.. contents::


Requirements
------------

This software requires Python >=3.8. To install the dependencies, run::

    poetry install

or::

    pip install -r requirements.txt

You are also free to use another package manager (e.g. Conda).

The `ARCA23K`__ and `FSD50K`__ datasets are required too. For
convenience, bash scripts are provided to download the datasets
automatically. The dependencies are bash, curl, and unzip. Simply run
the following command from the root directory of the project::

    $ scripts/download_arca23k.sh
    $ scripts/download_fsd50k.sh

This will download the datasets to a directory called ``_datasets/``.
When running the software, the ``--arca23k_dir`` and ``--fsd50k_dir``
options (refer to the `Usage`_ section) can be used to specify the
location of the datasets. This is only necessary if the dataset paths
are different from the default.

__ https://zenodo.org/record/5117901
__ https://zenodo.org/record/4060432


Usage
-----

The general usage pattern is::

    python <script> [-f PATH] <args...> [options...]

The command-line options can also be specified in configuration files.
The path of a configuration file can be specified to the program using
the ``--config_file`` (or ``-f``) command-line option. This option can
be used multiple times. Options that are passed in the command-line
override those in the config file(s). See `default.ini`__ for an example
of a config file. Note that default.ini does not need to be specified in
the command line and should not be modified.

__ default.ini


Training
^^^^^^^^

To train a model, run::

    python baseline/train.py DATASET [-f FILE] [--experiment_id ID] [--work_dir DIR] [--arca23k_dir DIR] [--fsd50k_dir DIR] [--frac NUM] [--sample_rate NUM] [--block_length NUM] [--hop_length NUM] [--features SPEC] [--cache_features BOOL] [--model {vgg9a,vgg11a}] [--weights_path PATH] [--label_noise DICT] [--n_epochs N] [--batch_size N] [--lr NUM] [--lr_scheduler SPEC] [--partition SPEC] [--seed N] [--cuda BOOL] [--n_workers N] [--overwrite BOOL]

The ``DATASET`` argument accepts the following values:

* ``arca23k`` - Train using the ARCA23K dataset.
* ``arca23k-fsd`` - Train using the ARCA23K-FSD dataset.
* ``arca23k-<p>`` - Train using a mixture of ARCA23K and ARCA23K-FSD.
  Replace ``<p>`` with a fraction that represents the percentage of
  ARCA23K examples to be present in the training set.

The ``--experiment_id`` option is used to differentiate experiments. It
determines where the output files are saved relative to the path given
by the ``--work_dir`` option. When running multiple trials, either use
the ``--seed`` option to specify different random seeds or set it to a
negative number to disable setting the random seed. Otherwise, the
learned models will be identical across different trials.

Example::

    python baseline/train.py arca23k --experiment_id my_experiment


Prediction
^^^^^^^^^^

To compute predictions, run::

    python baseline/predict.py DATASET SUBSET [-f FILE] [--experiment_id ID] [--work_dir DIR] [--arca23k_dir DIR] [--fsd50k_dir DIR] [--output_name FILE_NAME] [--clean BOOL] [--sample_rate NUM] [--block_length NUM] [--features SPEC] [--cache_features BOOL] [--weights_path PATH] [--batch_size N] [--partition SPEC] [--n_workers N] [--seed N] [--cuda BOOL]

The ``SUBSET`` argument must be set to either ``training``,
``validation``, or ``test``.

Example::

    python baseline/predict.py arca23k test --experiment_id my_experiment


Evaluation
^^^^^^^^^^

To evaluate the predictions, run::

    python baseline/evaluate.py DATASET SUBSET [-f FILE] [--experiment_id LIST] [--work_dir DIR] [--arca23k_dir DIR] [--fsd50k_dir DIR] [--output_name FILE_NAME] [--cached BOOL]

The ``SUBSET`` argument must be set to either ``training``,
``validation``, or ``test``.

Example::

    python baseline/evaluate.py arca23k test --experiment_id my_experiment


Citing
------

If you wish to cite this work, please cite the following paper:

.. [1] \T. Iqbal, Y. Cao, A. Bailey, M. D. Plumbley, and W. Wang,
       “ARCA23K: An audio dataset for investigating open-set label
       noise”, in Proceedings of the Detection and Classification of
       Acoustic Scenes and Events 2021 Workshop (DCASE2021), 2021,
       Barcelona, Spain, pp. 201–205.

BibTeX::

    @inproceedings{Iqbal2021,
        author = {Iqbal, T. and Cao, Y. and Bailey, A. and Plumbley, M. D. and Wang, W.},
        title = {{ARCA23K}: An audio dataset for investigating open-set label noise},
        booktitle = {Proceedings of the Detection and Classification of Acoustic Scenes and Events 2021 Workshop (DCASE2021)},
        pages = {201--205},
        year = {2021},
        address = {Barcelona, Spain},
    }
