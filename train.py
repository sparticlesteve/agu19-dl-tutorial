"""
Main training script for the distributed examples.
"""

# System imports
import os
import argparse
import logging

# External imports
import yaml
import numpy as np
import tensorflow as tf
# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(logging.ERROR)
import horovod.tensorflow.keras as hvd

# Local imports
from data import get_datasets
from models import get_model
from utils.optimizers import get_optimizer
from utils.callbacks import TimingCallback
from utils.device import configure_session
from utils.metrics import PixelAccuracy, PixelIoU

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('train.py')
    add_arg = parser.add_argument
    add_arg('config', nargs='?', default='configs/test.yaml')
    add_arg('-d', '--distributed', action='store_true')
    add_arg('-v', '--verbose', action='store_true')
    return parser.parse_args()

def config_logging(verbose):
    log_format = '%(asctime)s %(levelname)s %(message)s'
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format=log_format)

def init_workers(distributed=False):
    """Initialize distributed worker"""
    rank, local_rank, n_ranks = 0, 0, 1
    if distributed:
        hvd.init()
        rank, local_rank, n_ranks = hvd.rank(), hvd.local_rank(), hvd.size()
    return rank, local_rank, n_ranks

def load_config(config_file):
    """Load config from specified YAML file"""
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # Expand some environment variables
    config['output_dir'] = os.path.expandvars(config['output_dir'])
    return config

def main():
    """Main function"""

    # Initialization
    args = parse_args()
    rank, local_rank, n_ranks = init_workers(args.distributed)

    # Load configuration
    config = load_config(args.config)

    # Configure logging
    config_logging(verbose=args.verbose)
    logging.info('Initialized rank %i local_rank %i size %i',
                 rank, local_rank, n_ranks)

    # Device configuration
    configure_session(gpu=local_rank, **config.get('device', {}))

    # Load the data
    train_data, valid_data = get_datasets(rank=rank, n_ranks=n_ranks,
                                          **config['data'])
    if rank == 0:
        logging.info(train_data)
        logging.info(valid_data)

    # Construct the model and optimizer
    model = get_model(**config['model'])
    optimizer = get_optimizer(n_ranks=n_ranks, **config['optimizer'])
    train_config = config['train']

    # Custom metrics for pixel accuracy and IoU
    metrics = [PixelAccuracy(), PixelIoU(name='iou', num_classes=3)]

    # Compile the model
    model.compile(loss=train_config['loss'], optimizer=optimizer,
                  metrics=metrics)

    # Print a model summary
    if rank == 0:
        model.summary()

    # Prepare the callbacks
    callbacks = []

    if args.distributed:

        # Broadcast initial variable states from rank 0 to all processes.
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))

        # Average metrics across workers
        callbacks.append(hvd.callbacks.MetricAverageCallback())

        # Learning rate warmup
        warmup_epochs = train_config.get('lr_warmup_epochs', 0)
        callbacks.append(hvd.callbacks.LearningRateWarmupCallback(
            warmup_epochs=warmup_epochs, verbose=1))

    # Timing
    timing_callback = TimingCallback()
    callbacks.append(timing_callback)

    # Checkpointing and CSV logging from rank 0 only
    #if rank == 0:
    #    callbacks.append(tf.keras.callbacks.ModelCheckpoint(checkpoint_format))
    #    callbacks.append(tf.keras.callbacks.CSVLogger(
    #        os.path.join(config['output_dir'], 'history.csv'), append=args.resume))

    if rank == 0:
        logging.debug('Callbacks: %s', callbacks)

    # TEST class weights - these seem to be ignored no matter what!
    #class_weights = [0., 6., .3]
    #class_weights = {0 : 10000.02, 1 : 1.3e6, 2 : 8.2e2}
    #class_weights = [0.0217966, 0.81204625, 0.16615715]

    # Train the model
    verbosity = 2 if (rank==0 or args.verbose) else 0
    history = model.fit(train_data,
                        validation_data=valid_data,
                        epochs=train_config['n_epochs'],
                        class_weight=class_weights,
                        callbacks=callbacks,
                        verbose=verbosity)

    # All done
    if rank == 0:
        logging.info('All done!')

if __name__ == '__main__':
    main()
