from multiprocessing.sharedctypes import Value
from typing import Dict, List, Optional
import torch
from mmcv.runner import OPTIMIZER_BUILDERS, DefaultOptimizerConstructor
from mmcv.utils import SyncBatchNorm, _BatchNorm, _ConvNd, build_from_cfg
from mmcv.runner import OPTIMIZERS

import torch
import torch.nn as nn

@OPTIMIZER_BUILDERS.register_module()
class CombinedOptimizerConstructor:
    def __init__(self,
                 optimizer_cfg: Dict,
                 paramwise_cfg: Optional[Dict] = None):

        if paramwise_cfg is not None:
            raise ValueError('CombinedOptimizerConstructor cannot handle paramwise cfg'
                             ', please pass paramwise_cfg=None or use another constructor.'
                             )
        self.optimizer_cfg = optimizer_cfg

    def __call__(self, model: nn.Module):
        if hasattr(model, 'module'):
            model = model.module

        optimizer_cfg = self.optimizer_cfg.copy()


        grouped_optimizer_cfg = dict()
        grouped_optimizer_cfg['type'] = optimizer_cfg['type']
        grouped_optimizer_cfg['params'] = []

        video_settings = optimizer_cfg['video']
        video_module = getattr(model,optimizer_cfg["video_module_name"])
        video_params = video_module.parameters()
        grouped_optimizer_cfg['params'].append({"params": video_params, **video_settings})

        audio_settings = optimizer_cfg['audio']
        audio_module = getattr(model, optimizer_cfg["audio_module_name"])
        audio_params = audio_module.parameters()
        grouped_optimizer_cfg['params'].append({"params": audio_params, **audio_settings})

        head_settings = optimizer_cfg['head']
        head_module = getattr(model ,optimizer_cfg["head_module_name"])
        head_params = head_module.parameters()
        grouped_optimizer_cfg['params'].append({"params": head_params, **head_settings})
        

        return build_from_cfg(grouped_optimizer_cfg, OPTIMIZERS)

        