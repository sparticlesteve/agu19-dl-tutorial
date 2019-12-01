"""TF datasets"""

import importlib

def get_datasets(name, **data_args):
    """Import the dataset model module by name and call build_model"""
    module = importlib.import_module('.'+name, 'data')
    return module.get_datasets(**data_args)
