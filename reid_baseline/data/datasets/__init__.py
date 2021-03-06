# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
from .cuhk03 import CUHK03
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .aic_track2 import aic_track2
from .dataset_loader import ImageDataset

__factory = {
    'market1501': Market1501,
    'cuhk03': CUHK03,
    'dukemtmc': DukeMTMCreID,
    'aic_track2': aic_track2
}


def get_names():
    return __factory.keys()


def init_dataset(name, cfg, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](cfg, *args, **kwargs)
