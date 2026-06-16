import os
import tempfile
import unittest

import mmcv
import numpy as np

from projects.mmdet3d_plugin.datasets.pipelines.loading import (
    SimV2ILoadMultiViewImageFromFiles,
)
from projects.mmdet3d_plugin.datasets.pipelines.transform_3d import (
    NormalizeMultiviewImage,
)


class SimV2IImageLoadingTest(unittest.TestCase):

    def test_rgba_image_is_loaded_as_three_channels(self):
        rgba = np.zeros((8, 12, 4), dtype=np.uint8)
        rgba[..., 0] = 10
        rgba[..., 1] = 20
        rgba[..., 2] = 30
        rgba[..., 3] = 255

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, 'camera.png')
            self.assertTrue(mmcv.imwrite(rgba, image_path))
            raw = mmcv.imread(image_path, flag='unchanged')
            self.assertEqual(raw.shape, (8, 12, 4))

            loader = SimV2ILoadMultiViewImageFromFiles(to_float32=True)
            results = loader({'img_filename': [image_path]})
            self.assertEqual(results['img'][0].shape, (8, 12, 3))
            self.assertEqual(results['img_shape'], (8, 12, 3, 1))
            self.assertEqual(results['img'][0].dtype, np.float32)

            normalize = NormalizeMultiviewImage(
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True,
            )
            normalized = normalize(results)
            self.assertEqual(normalized['img'][0].shape, (8, 12, 3))


if __name__ == '__main__':
    unittest.main()
