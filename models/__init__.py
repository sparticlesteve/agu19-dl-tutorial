"""Keras model functions"""

import importlib

def get_model(name, **model_args):
    """Import the model module by name and call build_model"""
    module = importlib.import_module('.'+name, 'models')
    return module.build_model(**model_args)
