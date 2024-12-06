import numpy as np
import torch

from .data_process import DataProcess2


class NormalizeImage(DataProcess2):
    RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])

    def process(self, data):
        assert 'image' in data, '`image` in data is required by this process'
        #data['qzz'] = data['image'].copy()
        # print(type(data['image']))
        # raise
        image = data['image']
        image -= self.RGB_MEAN
        image /= 255.
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        data['image'] = image
        return data

    @classmethod
    def restore(self, image):
        image = image.permute(1, 2, 0).to('cpu').numpy()
        image = image * 255.
        image += self.RGB_MEAN
        image = image.astype(np.uint8)
        return image
