# Copyright (c) OpenMMLab. All rights reserved.

from mmcv.utils import Registry

from .temporal_encoder import TemporalGRUEncoder
from .tc_gap import TCGap

NECKS = Registry('necks')

NECKS.register_module(name='TemporalGRUEncoder', module=TemporalGRUEncoder)
NECKS.register_module(name='TCGap', module=TCGap)


def build_neck(cfg):
    """Build neck."""
    if cfg is None:
        return None
    return NECKS.build(cfg)
