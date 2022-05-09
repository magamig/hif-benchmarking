"""
@Author: zhengke
@Date: 2020-07-20 08:04:14
@LastEditTime: 2020-07-20 08:42:31
@LastEditors: zhengke
@Description: 
@FilePath: \DeformatedHyperImageFusion\model\__init__.py
"""

import importlib
from model.base_model import BaseModel


def get_option_setter(model_name):
    """
    @description: transfer the network parser options
    @param {model name} 
    @return: static function of network parser
    """
    model_class = find_model_using_name(model_name)
    return model_class.modify_commandline_options


def find_model_using_name(model_name):
    """
    @description: return an instanced class according the model's name,
                    the model should be a subclass of basedmodel
    @param {model name} 
    @return: instanced network class
    """
    model_filename = "model." + model_name
    modellib = importlib.import_module(model_filename)

    model = None
    target_model_name = model_name.replace("_", "")
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() and issubclass(cls, BaseModel):
            model = cls

    if model is None:
        print(
            "In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase."
            % (model_filename, target_model_name)
        )
        exit(0)

    return model


def create_model(opt, data_dict):
    """
    @description: like the name, to create a instanced model class
    @param {parser options, data_dict} 
    @return: instanced model class
    """
    model_class = find_model_using_name(opt.model_name)
    instance = model_class()
    instance.initialize(opt, data_dict)
    return instance
