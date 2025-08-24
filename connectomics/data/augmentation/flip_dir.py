from __future__ import print_function, division
from typing import Optional
from skimage import io
import numpy as np
from .augmentor import DataAugment

class Flip(DataAugment):
    r"""
    Randomly flip along `z`-, `y`- and `x`-axes as well as swap `y`- and `x`-axes
    for anisotropic image volumes. For learning on isotropic image volumes set
    :attr:`do_ztrans` to 1 to swap `z`- and `x`-axes (the inputs need to be cubic).
    This augmentation is applied to both images and masks.

    Args:
        do_ztrans (int): set to 1 to swap z- and x-axes for isotropic data. Default: 0
        p (float): probability of applying the augmentation. Default: 0.5
        additional_targets(dict, optional): additional targets to augment. Default: None
    """
    def __init__(self,

                 p: float = 0.5,
                 additional_targets: Optional[dict] = None,
                 skip_targets: list = []):

        super(Flip, self).__init__(p, additional_targets, skip_targets)


    def set_params(self):
        r"""There is no change in sample size.
        """
        pass

    def flip_and_swap(self, data, rule):
        assert data.ndim==3
        if data.ndim == 3: # 3-channel input in z,y,x
            # z reflection.
            if rule[0]:
                data = data[::-1, :, :]
            # y reflection.
            if rule[1]:
                data = data[:, ::-1, :]
            # x reflection.
            if rule[2]:
                data = data[:, :, ::-1]
            # Transpose in xy.
            if rule[3]:
                data = data.transpose(0, 2, 1)


        return data

    def __call__(self, sample, random_state=np.random.RandomState()):
        rule = random_state.randint(2, size=4)
        #rule=[0,0,0,1]
        '''
        n = np.random.randint(0, 5)
        dir1=np.array([sample['dz'],sample['dy'],sample['dx']])
        tif1=dir1.transpose( (1,2,3,0))
        ind = np.where(tif1 < 0)
        tif1[ind] *= -1
        tif1 *= 255
        tif1 = np.array(tif1, dtype='uint8')
        str_rule=str(rule[0]) +str(rule[1]) +str(rule[2]) +str(rule[3])
        io.imsave('/home/yanhy/share/a' +str(n) + '.tif', tif1)
        '''



        sample['image'] = self.flip_and_swap(sample['image'].copy(), rule)
        for key in self.additional_targets.keys():
            if key not in self.skip_targets:
                sample[key] = self.flip_and_swap(sample[key].copy(), rule)
        #dir
        if rule[0]:
            sample['dz']=-sample['dz']
        # y reflection.
        if rule[1]:
            sample['dy']=-sample['dy']
        # x reflection.
        if rule[2]:
            sample['dx']=-sample['dx']
        # Transpose in xy.
        if rule[3]:
            sample["dx"], sample["dy"] = sample["dy"], sample["dx"]

       

        return sample
