from __future__ import print_function, division
from typing import Union, List

import os
import math
import glob
import copy
import numpy as np
from scipy.ndimage import zoom
from skimage.transform import resize
    
import h5py
import torch
import torch.utils.data

from .dataset_volume import VolumeDataset, VolumeDatasetMultiSeg
from .dataset_tile import TileDataset
from .collate import *
from ..utils import *


def _make_path_list(cfg, dir_name, file_name, rank=None):
    r"""Concatenate directory path(s) and filenames and return
    the complete file paths.
    """
    if not cfg.DATASET.IS_ABSOLUTE_PATH:
        assert len(dir_name) == 1 or len(dir_name) == len(file_name)
        if len(dir_name) == 1:
            file_name = [os.path.join(dir_name[0], x) for x in file_name]
        else:
            file_name = [os.path.join(dir_name[i], file_name[i])
                         for i in range(len(file_name))]

        if cfg.DATASET.LOAD_2D: # load 2d images
            temp_list = copy.deepcopy(file_name)
            file_name = []
            for x in temp_list:
                suffix = x.split('/')[-1]
                if suffix in ['*.png', '*.tif']:
                    file_name += sorted(glob.glob(x, recursive=True))
                else: # complete filename is specified
                    file_name.append(x)

    file_name = _distribute_data(cfg, file_name, rank)
    return file_name


def _distribute_data(cfg, file_name, rank=None):
    r"""Distribute the data (files) equally for multiprocessing.
    """
    if rank is None or cfg.DATASET.DISTRIBUTED == False:
        return file_name

    world_size = cfg.SYSTEM.NUM_GPUS
    num_files = len(file_name)
    ratio = num_files / float(world_size)
    ratio = int(math.ceil(ratio-1) + 1)  # 1.0 -> 1, 1.1 -> 2

    extended = [file_name[i % num_files] for i in range(world_size*ratio)]
    splited = [extended[i:i+ratio] for i in range(0, len(extended), ratio)]

    return splited[rank]


def _get_file_list(name: Union[str, List[str]],
                   prefix: Optional[str] = None) -> list:
    # name: datasets/SNEMI3D/
    # prefix :前缀
    if isinstance(name, list):
        return name

    suffix = name.split('.')[-1]#后缀
    if suffix == 'txt':  # a text file saving the absolute path
        filelist = [line.rstrip('\n') for line in open(name)]
        return filelist

    suffix = name.split('/')[-1]  #SNEMI3D
    if suffix in ['*.png', '*.tif']: # find all image files under a folder
        assert prefix is not None
        filelist = sorted(glob.glob(os.path.join(
            prefix, name), recursive=True))
        return [os.path.relpath(x, prefix) for x in filelist]

    return name.split('@')


def _rescale(data: np.ndarray, scales: List[float], order: int):
    if scales is not None and (np.array(scales) != 1).any():
        if data.ndim == 3:
            return zoom(data, scales, order=order)

        assert data.ndim == 4 # c,z,y,x
        n_maps = data.shape[0]
        return np.stack([
            zoom(data[i], scales, order=order) for i in range(n_maps)
        ], 0)

    return data # no rescaling


def _pad(data: np.ndarray, pad_size: Union[List[int], int], pad_mode: str):
    pad_size = get_padsize(pad_size)
    if data.ndim == 3:
        return np.pad(data, pad_size, pad_mode)

    assert data.ndim == 4 # c,z,y,x
    pad_size = [(0, 0)] + list(pad_size) # no padding for channel dim
    return np.pad(data, tuple(pad_size), pad_mode)


def _resize2target(data: np.ndarray, enabled: bool = False, order: int = 0,
                   target_size: Optional[tuple] = None):
    """If the data is not larger than or equal to the target size in
    all dimensions, resize to the minimal size adequate for sampling.
    """
    if (not enabled) or (target_size is None):
        return data

    # data should be in (z,y,x) or (c,z,y,x) formats
    assert data.ndim in [3, 4]
    data_size, target_size = np.array(data.shape), np.array(target_size)
    if (data_size[-3:] >= target_size).all():
        return data # size is large enough for sampling
                
    dtype = data.dtype
    min_size = tuple(np.maximum(data_size[-3:], target_size))
    if data.ndim == 4:
        min_size = tuple(data.shape[0] + list(min_size)) # keep channel number
    data = resize(data, min_size, order=order, anti_aliasing=False, 
                  preserve_range=True).astype(dtype)

    return data


def _load_label_condition(name, mode: str, image_only_test: bool):
    condition0 = name is not None
    condition1 = mode in ['train', 'val']
    if image_only_test: # only load image during inference
        return condition0 and condition1
    
    # mask can also be loaded at inference time if not None
    return condition0
 


def build_micro():
    volume = []
    mask = []
    segmentation = []
    filePath = '/home/yanhy/pytorch_connectomics/dataset/cp/img/'
    sge_path = '/home/yanhy/pytorch_connectomics/dataset/segm_singlechannel/'
    mask_path = '/home/yanhy/pytorch_connectomics/dataset/cp/mask/'
    pic_list = os.listdir(filePath)
    for rby in pic_list:
      if '.npy' in rby:
        v = np.load(os.path.join(filePath, rby))
        segm = np.load(os.path.join(sge_path, rby))
        m = np.load(os.path.join(mask_path, rby))
        if 'pinky_vol503' in rby :
          if 'pinky_vol503' in rby:
            x=5
          else:
            x=2
          for i in range(x):
            segmentation.append(segm)
            volume.append(v)
            mask.append(m)
            
          print('repute!')
          
          
        
        segmentation.append(segm)
        volume.append(v)
        print(rby,v.shape)
        print(np.unique(segm,return_counts=True))
        mask.append(m)

    return volume, segmentation, mask


def get_dataset(cfg,
                augmentor,
                mode='train',
                rank=None,
                dataset_class=VolumeDataset,
                dataset_options={},
                dir_name_init: Optional[list] = None,
                img_name_init: Optional[list] = None):
    r"""Prepare dataset for training and inference.
    """
    assert mode in ['train', 'val', 'test']

    sample_label_size = cfg.MODEL.OUTPUT_SIZE #单个块大小
    topt, wopt = ['0'], [['0']]
    if mode == 'train':
        sample_volume_size = augmentor.sample_size if augmentor is not None else cfg.MODEL.INPUT_SIZE
      
        
        sample_label_size = sample_volume_size
        sample_stride = (1, 1, 1)
        topt, wopt = cfg.MODEL.TARGET_OPT, cfg.MODEL.WEIGHT_OPT#？？？？
        iter_num = cfg.SOLVER.ITERATION_TOTAL * cfg.SOLVER.SAMPLES_PER_BATCH
        if cfg.SOLVER.SWA.ENABLED:
            iter_num += cfg.SOLVER.SWA.BN_UPDATE_ITER

    elif mode == 'val':
        sample_volume_size = cfg.MODEL.INPUT_SIZE
        sample_label_size = sample_volume_size
        sample_stride = [x//2 for x in sample_volume_size]
        topt, wopt = cfg.MODEL.TARGET_OPT, cfg.MODEL.WEIGHT_OPT
        iter_num = -1

    elif mode == 'test':
        sample_volume_size = cfg.MODEL.INPUT_SIZE
        sample_stride = cfg.INFERENCE.STRIDE
        iter_num = -1

    shared_kwargs = {
        "sample_volume_size": sample_volume_size,
        "sample_label_size": sample_label_size,
        "sample_stride": sample_stride,
        "augmentor": augmentor,
        "target_opt": topt,#要学习的目标列表
        "weight_opt": wopt,#loss是否有效
        "mode": mode,
        "do_2d": cfg.DATASET.DO_2D,
        "reject_size_thres": cfg.DATASET.REJECT_SAMPLING.SIZE_THRES,
        "reject_diversity": cfg.DATASET.REJECT_SAMPLING.DIVERSITY,
        "reject_p": cfg.DATASET.REJECT_SAMPLING.P,
        "data_mean": cfg.DATASET.MEAN,
        "data_std": cfg.DATASET.STD,
        "data_match_act": cfg.DATASET.MATCH_ACT,
        "erosion_rates": cfg.MODEL.LABEL_EROSION,
        "dilation_rates": cfg.MODEL.LABEL_DILATION,
        "do_relabel": cfg.DATASET.REDUCE_LABEL,
        "valid_ratio": cfg.DATASET.VALID_RATIO,
    }

    if cfg.DATASET.DO_CHUNK_TITLE == 1:  # build TileDataset
        def _make_json_path(path, name):
            if isinstance(name, str):
                return [os.path.join(path, name)]

            assert isinstance(name, (list, tuple))
            json_list = [os.path.join(path, name[i]) for i in range(len(name))]
            return json_list

        input_path = cfg.DATASET.INPUT_PATH
        volume_json = _make_json_path(input_path, cfg.DATASET.IMAGE_NAME)

        label_json, valid_mask_json = None, None
        if mode == 'train':
            if cfg.DATASET.LABEL_NAME is not None:
                label_json = _make_json_path(input_path, cfg.DATASET.LABEL_NAME)
            if cfg.DATASET.VALID_MASK_NAME is not None:
                valid_mask_json = _make_json_path(input_path, cfg.DATASET.VALID_MASK_NAME)

        dataset = TileDataset(chunk_num=cfg.DATASET.DATA_CHUNK_NUM,
                              chunk_ind=cfg.DATASET.DATA_CHUNK_IND,
                              chunk_ind_split=cfg.DATASET.CHUNK_IND_SPLIT,
                              chunk_iter=cfg.DATASET.DATA_CHUNK_ITER,
                              chunk_stride=cfg.DATASET.DATA_CHUNK_STRIDE,
                              volume_json=volume_json,
                              label_json=label_json,
                              valid_mask_json=valid_mask_json,
                              pad_size=cfg.DATASET.PAD_SIZE,
                              data_scale=cfg.DATASET.DATA_SCALE,
                              coord_range=cfg.DATASET.DATA_COORD_RANGE,
                              **shared_kwargs)

    else:  # build VolumeDataset or VolumeDatasetMultiSeg
        volume, label, valid_mask = build_micro()
        

            #_get_input(
            #cfg, mode, rank, dir_name_init, img_name_init, min_size=sample_volume_size)

        if cfg.MODEL.TARGET_OPT_MULTISEG_SPLIT is not None:
            shared_kwargs['multiseg_split'] = cfg.MODEL.TARGET_OPT_MULTISEG_SPLIT
        dataset = dataset_class(volume=volume, label=label, valid_mask=valid_mask,
                                iter_num=iter_num, **shared_kwargs, **dataset_options)

    return dataset


def build_dataloader(cfg, augmentor=None, mode='train', dataset=None, rank=None,
                     dataset_class=VolumeDataset, dataset_options={}, cf=collate_fn_train):
    r"""Prepare dataloader for training and inference.
    """
    assert mode in ['train', 'val', 'test']
    print('Mode: ', mode)

    if mode == 'train':
        batch_size = cfg.SOLVER.SAMPLES_PER_BATCH
    elif mode == 'val':
        batch_size = cfg.SOLVER.SAMPLES_PER_BATCH * 4
    else:
        cf = collate_fn_test # update the collate function
        batch_size = cfg.INFERENCE.SAMPLES_PER_BATCH * cfg.SYSTEM.NUM_GPUS

    if dataset is None: # no pre-defined dataset instance
        if cfg.MODEL.TARGET_OPT_MULTISEG_SPLIT is not None:
            dataset_class = VolumeDatasetMultiSeg
        dataset = get_dataset(cfg, augmentor, mode, rank, dataset_class, dataset_options)

    sampler = None
    num_workers = cfg.SYSTEM.NUM_CPUS
    if cfg.SYSTEM.DISTRIBUTED:
        num_workers = cfg.SYSTEM.NUM_CPUS // cfg.SYSTEM.NUM_GPUS
        if cfg.DATASET.DISTRIBUTED == False:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    # In PyTorch, each worker will create a copy of the Dataset, so if the data
    # is preload the data, the memory usage should increase a lot.
    # https://discuss.pytorch.org/t/define-iterator-on-dataloader-is-very-slow/52238/2
    img_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=cf,
        sampler=sampler, num_workers=num_workers, pin_memory=True)

    return img_loader




