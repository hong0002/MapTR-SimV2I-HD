from .nuscenes_dataset import CustomNuScenesDataset
from .builder import custom_build_dataset

from .nuscenes_map_dataset import CustomNuScenesLocalMapDataset
from .simv2i_map_dataset import SimV2IMapDataset
__all__ = [
    'CustomNuScenesDataset', 'CustomNuScenesLocalMapDataset',
    'SimV2IMapDataset'
]

# AV2 is optional and its current package releases require a newer NumPy stack.
try:
    from .av2_map_dataset import CustomAV2LocalMapDataset
except (ImportError, TypeError):
    CustomAV2LocalMapDataset = None
else:
    __all__.append('CustomAV2LocalMapDataset')
