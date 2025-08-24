from typing import Optional, List
import numpy as np
import random

import torch
import torch.utils.data
from ..augmentation import Compose
from ..utils import *
from skimage.measure import block_reduce
from skimage import io
import copy
import time

TARGET_OPT_TYPE = List[str]
WEIGHT_OPT_TYPE = List[List[str]]
AUGMENTOR_TYPE = Optional[Compose]


class VolumeDataset(torch.utils.data.Dataset):
    background: int = 0  # background label index

    def __init__(self,
                 volume: list,
                 label: Optional[list] = None,
                 valid_mask: Optional[list] = None,
                 points: Optional[list] = None,
                 valid_ratio: float = 0.5,
                 sample_volume_size: tuple = (8, 64, 64),
                 sample_label_size: tuple = (8, 64, 64),
                 sample_stride: tuple = (1, 1, 1),
                 augmentor: AUGMENTOR_TYPE = None,
                 target_opt: TARGET_OPT_TYPE = ['1'],
                 weight_opt: WEIGHT_OPT_TYPE = [['1']],
                 erosion_rates: Optional[List[int]] = None,
                 dilation_rates: Optional[List[int]] = None,
                 mode: str = 'train',
                 do_2d: bool = False,
                 iter_num: int = -1,
                 do_relabel: bool = True,
                 # rejection sampling
                 reject_size_thres: int = 0,
                 reject_diversity: int = 0,
                 reject_p: float = 0.95,
                 # normalization
                 data_mean: float = 0.5,
                 data_std: float = 0.5,
                 data_match_act: str = 'none'):

        assert mode in ['train', 'val', 'test']
        self.points = points

        self.mode = mode
        self.do_2d = do_2d
        self.do_relabel = False

        # data format
        self.volume = volume
        self.label = label

        self.augmentor = augmentor

        # target and weight options
        self.target_opt = target_opt
        self.weight_opt = weight_opt
        # For 'all', users will create their own targets
        if self.target_opt[-1] == 'all':
            self.target_opt = self.target_opt[:-1]
            self.weight_opt = self.weight_opt[:-1]
        self.erosion_rates = erosion_rates
        self.dilation_rates = dilation_rates

        # rejection samping
        self.reject_size_thres = reject_size_thres
        self.reject_diversity = reject_diversity
        self.reject_p = reject_p

        # normalization
        self.data_mean = data_mean
        self.data_std = data_std
        self.data_match_act = data_match_act

        # dataset: channels, depths, rows, cols
        # volume size, could be multi-volume input
        self.volume_size = [np.array(x.shape) for x in self.volume]
        self.sample_volume_size = np.array(sample_volume_size).astype(int)  # model input size

        if self.label is not None:
            self.sample_label_size = np.array(
                sample_label_size).astype(int)  # model label size
            self.label_vol_ratio = self.sample_label_size / self.sample_volume_size
            if self.augmentor is not None:
                assert np.array_equal(
                    self.augmentor.sample_size, self.sample_label_size)
        self._assert_valid_shape()

        # compute number of samples for each dataset (multi-volume input)
        self.sample_stride = np.array(sample_stride).astype(int)
        self.sample_size = [count_volume(self.volume_size[x], self.sample_volume_size, self.sample_stride)
                            for x in range(len(self.volume_size))]

        # total number of possible inputs for each volume
        self.sample_num = np.array([np.prod(x) for x in self.sample_size])
        self.sample_num_a = np.sum(self.sample_num)
        self.sample_num_c = np.cumsum([0] + list(self.sample_num))

        # handle partially labeled volume
        self.valid_mask = valid_mask
        self.valid_ratio = valid_ratio

        if self.mode in ['val', 'test']:  # for validation and test
            self.sample_size_test = [
                np.array([np.prod(x[1:3]), x[2]]) for x in self.sample_size]

        # For relatively small volumes, the total number of samples can be generated is smaller
        # than the number of samples required for training (i.e., iteration * batch size). Thus
        # we let the __len__() of the dataset return the larger value among the two during training.
        self.iter_num = max(
            iter_num, self.sample_num_a) if self.mode == 'train' else self.sample_num_a
        print('Total number of samples to be generated: ', self.iter_num)

    def __len__(self):
        # total number of possible samples
        return self.iter_num

    def __getitem__(self, index):
        # orig input: keep uint/int format to save cpu memory
        # output sample: need np.float32

        vol_size = np.array(self.sample_volume_size)
        if self.mode == 'train':
            vol_size = np.array(self.sample_volume_size) // 3
            sample = self._rejection_sampling(vol_size)
            return self._process_targets(sample)

        elif self.mode == 'val':
            pos = self._get_pos_test(index)
            sample = self._crop_with_pos(pos, vol_size)
            return self._process_targets(sample)

        elif self.mode == 'test':

            pos = self._get_pos_test(index)
            # print(vol_size,'vol_size')

            out_volume = (crop_volume_l(
                self.volume[pos[0]], vol_size, pos[1:]) / 255.0).astype(np.float32)

            D, W, H = out_volume.shape[0], out_volume.shape[1], out_volume.shape[2]

            volume_ROI = out_volume[D // 3: 2 * D // 3, W // 3: 2 * W // 3, H // 3: 2 * H // 3]
            volume_ROI = self._process_image(volume_ROI)
            out_volume = block_reduce(out_volume, (3, 3, 3), func=np.max)
            out_volume = self._process_image(out_volume)

            return pos, volume_ROI, out_volume

    def _process_targets(self, sample):

        pos, out_volume, out_label_l, out_valid = sample
        D, W, H = out_volume.shape[0], out_volume.shape[1], out_volume.shape[2]

        volume_ROI = out_volume[D // 3: 2 * D // 3, W // 3: 2 * W // 3, H // 3: 2 * H // 3]
        volume_ROI = self._process_image(volume_ROI)

        out_volume = block_reduce(out_volume, (3, 3, 3), func=np.max)
        out_volume = self._process_image(out_volume)

        out_label_s = copy.deepcopy(out_label_l[D // 3: 2 * D // 3, W // 3: 2 * W // 3, H // 3: 2 * H // 3])
        out_label_l = block_reduce(out_label_l, (3, 3, 3), func=np.max)

        out_target_l = seg_to_targets(
            out_label_l, self.target_opt, self.erosion_rates, self.dilation_rates)
        out_target_s = seg_to_targets(
            out_label_s, self.target_opt, self.erosion_rates, self.dilation_rates)

        out_weight = [[np.array([0])]]

        return pos, volume_ROI, out_target_l, out_target_s, out_weight, out_volume

    #######################################################
    # Position Calculator
    #######################################################

    def _index_to_dataset(self, index):
        return np.argmax(index < self.sample_num_c) - 1  # which dataset

    def _index_to_location(self, index, sz):
        # index -> z,y,x
        # sz: [y*x, x]
        pos = [0, 0, 0]
        pos[0] = np.floor(index / sz[0])
        pz_r = index % sz[0]
        pos[1] = int(np.floor(pz_r / sz[1]))
        pos[2] = pz_r % sz[1]
        return pos

    def _get_pos_test(self, index):
        pos = [0, 0, 0, 0]
        did = self._index_to_dataset(index)
        pos[0] = did
        index2 = index - self.sample_num_c[did]
        pos[1:] = self._index_to_location(index2, self.sample_size_test[did])
        # if out-of-bound, tuck in
        for i in range(1, 4):
            if pos[i] != self.sample_size[pos[0]][i - 1] - 1:
                pos[i] = int(pos[i] * self.sample_stride[i - 1])
            else:
                pos[i] = int(self.volume_size[pos[0]][i - 1] -
                             self.sample_volume_size[i - 1])
        return pos

    def _get_pos_train(self, vol_size):
        # random: multithread
        # np.random: same seed

        pos = [0, 0, 0, 0]
        did = self._index_to_dataset(random.randint(0, self.sample_num_a - 1))

        pos[0] = did

        tmp_size = count_volume(  # 计算出有多少位置可以采样，返回[z,y,x]
            self.volume_size[did], vol_size, self.sample_stride)


        tmp_pos = [random.randint(0, tmp_size[x] - 1) * self.sample_stride[x]
                   for x in range(len(tmp_size))]

        pos[1:] = tmp_pos
        return pos

    #######################################################
    # Volume Sampler
    #######################################################
    def _rejection_sampling(self, vol_size):
        sample_count = 0
        while True:
            pos = self._get_pos_train(vol_size)
            pos, out_volume, out_label, out_valid = self._crop_with_pos(pos, vol_size)
            if self._is_fg(out_label):
                pos, out_volume, out_label, out_valid = self._crop_with_pos_l(pos, vol_size)
                data = {'image': out_volume, 'label': out_label}
                augmented = self.augmentor(data)
                out_volume, out_label = augmented['image'], augmented['label']
                return pos, out_volume, out_label, out_valid
            sample_count += 1

    def _crop_with_pos(self, pos, vol_size):
        out_valid = None
        out_volume = (crop_volume(
            self.volume[pos[0]], vol_size, pos[1:]) / 255.0).astype(np.float32)
        out_label = (crop_volume(self.label[pos[0]], vol_size, pos[1:])).astype(np.float32)
        return pos, out_volume, out_label, out_valid

    def _crop_with_pos_l(self, pos, vol_size):
        out_valid = None
        out_volume = (crop_volume_l(self.volume[pos[0]], vol_size, pos[1:]) / 255.0).astype(np.float32)
        out_label  = (crop_volume_l(self.label[pos[0]], vol_size, pos[1:])).astype(np.float32)

        return pos, out_volume, out_label, out_valid

    def _is_fg(self, out_label: np.ndarray) -> bool:
        p = self.reject_p
        size_thres = self.reject_size_thres
        if out_label.sum() < size_thres and random.random() < p:
                return False
        return True

    #######################################################
    # Utils
    #######################################################
    def _process_image(self, x: np.array):
        x = np.expand_dims(x, 0)  # (z,y,x) -> (c,z,y,x)
        x = normalize_image(x, self.data_mean, self.data_std,
                            match_act=self.data_match_act)
        return x

    def _assert_valid_shape(self):

        print(self.sample_volume_size)
        assert all(
            [(self.sample_volume_size <= x * 3).all()
             for x in self.volume_size]
        ), "Input size should be smaller than volume size."

