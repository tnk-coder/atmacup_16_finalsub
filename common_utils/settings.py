import os
import torch
import numpy as np
import random


# 乱数固定
def set_seed(seed=None, cudnn_deterministic=True):
    if seed is None:
        seed = 42

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic  # A100,effnetだとFalseの方が早い
    torch.backends.cudnn.benchmark = False

def get_env_name() -> str:
    env_keys = os.environ.keys()

    if "KAGGLE_DOCKER_IMAGE" in env_keys:
        env_name = 'kaggle'
    elif "NV_CUDA_LIB_VERSION" in env_keys:  # elif "COLAB_GPU" in env_keys:
        env_name = 'colab'
    else:
        env_name = 'local'

    return env_name


"""
def set_env_name():
    CFG.env_name = get_env_name()
    print('env_name', CFG.env_name)
"""

"""
def set_dataset_path(cfg):
    # CFG.exp_name = os.path.dirname(__file__).split('/')[-2]

    # CFG.comp_dir_path = '/notebooks/'

    if not hasattr(cfg, 'comp_dataset_path'):
        cfg.comp_dataset_path = f'{cfg.comp_dir_path}datasets/{cfg.comp_name}/'
        print('comp_dataset_path')
        print(cfg.comp_dataset_path)

    cfg.outputs_path = cfg.comp_dir_path +  \
        f'outputs/{cfg.comp_name}/{cfg.exp_name}/'

    cfg.submission_dir = cfg.outputs_path + 'submissions/'
    cfg.submission_path = cfg.submission_dir + f'submission_{cfg.exp_name}.csv'

    cfg.model_dir = cfg.outputs_path + \
        f'{cfg.comp_name}-models/'

    cfg.figures_dir = cfg.outputs_path + 'figures/'

    cfg.log_dir = cfg.outputs_path + 'logs/'
    cfg.log_path = cfg.log_dir + f'{cfg.exp_name}.txt'
"""

def make_dirs(cfg):
    for dir in [cfg.model_dir, cfg.figures_dir, cfg.submission_dir, cfg.log_dir]:
        os.makedirs(dir, exist_ok=True)


def cfg_init(cfg, mode='train'):
    set_seed(cfg.seed)
    # set_env_name()
    # set_dataset_path(cfg)

    if mode == 'train':
        make_dirs(cfg)
