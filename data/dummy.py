"""Synthetic dummy dataset specification"""

# External imports
import tensorflow as tf

def construct_dataset(n_samples, input_shape, target_shape, batch_size):
    """Construct a dataset of inputs and targets"""
    x = tf.random.uniform([n_samples] + input_shape)
    y = tf.random.uniform([n_samples] + target_shape)
    data = tf.data.Dataset.from_tensor_slices((x, y))
    return data.batch(batch_size)

def get_datasets(input_shape, target_shape, n_train, n_valid, batch_size,
                 rank=0, n_ranks=1):
    """Construct synthetic datasets for training and validation"""
    train_dataset = construct_dataset(n_samples=n_train,
                                      input_shape=input_shape,
                                      target_shape=target_shape,
                                      batch_size=batch_size)
    valid_dataset = construct_dataset(n_samples=n_valid,
                                      input_shape=input_shape,
                                      target_shape=target_shape,
                                      batch_size=batch_size) if n_valid > 0 else None
    return train_dataset, valid_dataset
