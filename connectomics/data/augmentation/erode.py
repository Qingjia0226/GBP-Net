import numpy as np
from scipy.ndimage.morphology import grey_erosion
from typing import Optional
from .augmentor import DataAugment
from skimage import io
class Erode(DataAugment):
    r"""
    Erode the input image.

    Args:
        p (float): probability of applying the augmentation. Default: 0.5
        additional_targets(dict, optional): additional targets to augment. Default: None
    """

    def __init__(self,
                 p: float = 0.6,
                 additional_targets: Optional[dict] = None,
                 skip_targets: list = []):

        super(Erode, self).__init__(p, additional_targets, skip_targets)

    def set_params(self):
        pass
    
    def erode(self, imgs, labels, times):
        z, center_x, center_y=0,0,0
        D, W, H = imgs.shape
        
        transformed_imgs = np.copy(imgs[ D // 3: 2 * D // 3, W // 3: 2 * W // 3, H // 3: 2 * H // 3])
        positive_indices = np.where(labels == 1)  
        if len(positive_indices[0]) > 0:
            for jj in range(5):
              center_index = np.random.choice(len(positive_indices[0]))  
              z, center_x, center_y = positive_indices[0][center_index], positive_indices[1][center_index], \
                                    positive_indices[2][center_index]
            # 遍历中心点上下几层，在以选定正样本为中心的 10*20*20 范围内进行腐蚀
              try:
                  start_z, end_z = max(z - 5, 0), min(z + 6, imgs.shape[0])
                  for zz in range(start_z, end_z):
                    img = transformed_imgs[zz, center_x - 15:center_x + 15, center_y - 15:center_y + 15]
                    for i in range(times):
                        img = grey_erosion(img, size=(3, 3))
                    transformed_imgs[zz, center_x - 15:center_x + 15, center_y - 15:center_y + 15] = img
              except:
                   pass
        imgs[ D // 3: 2 * D // 3, W // 3: 2 * W // 3, H // 3: 2 * H // 3]=transformed_imgs
        return imgs
        
    def __call__(self, sample, random_state=np.random.RandomState()):
        images = sample['image'].copy()
        
        times = random_state.randint(1,3)
        num=random_state.randint(0, 100)
        
        sample['image']= self.erode(images, sample['label'],times)
        

        return sample
