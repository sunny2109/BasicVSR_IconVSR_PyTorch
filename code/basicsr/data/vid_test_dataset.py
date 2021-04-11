import glob
import torch
import os
from os import path as osp
from torch.utils import data as data

from basicsr.data.data_util import read_img_seq
from basicsr.utils import get_root_logger, scandir


class VidTestDataset(data.Dataset):
    """Vid4 test dataset.
        dataroot
        ├── subfolder1
            ├── frame000
            ├── frame001
            ├── ...
        ├── subfolder1
            ├── frame000
            ├── frame001
            ├── ...
        ├── ...

    For testing datasets, there is no need to prepare LMDB files.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
    """

    def __init__(self, opt):
        super(VidTestDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        assert self.io_backend_opt[
            'type'] != 'lmdb', 'No need to use lmdb during validation/test.'
        
        logger = get_root_logger()
        logger.info(f'Generate data info for VideoTestDataset - {opt["name"]}')

        self.data_info = {
            'lq_path': [],
            'gt_path': [],
            'clip_name': [],
            'max_idx': [],
        }
        self.lq_frames, self.gt_frames = {}, {}
                
        self.clip_list = os.listdir(osp.abspath(self.gt_root))
        self.clip_list.sort()
        for clip_name in self.clip_list:
            lq_frames_path = osp.join(self.lq_root, clip_name)
            lq_frames_path = sorted(
                    list(scandir(lq_frames_path, full_path=True)))
            
            gt_frames_path = osp.join(self.gt_root, clip_name)
            gt_frames_path = sorted(
                    list(scandir(gt_frames_path, full_path=True)))

            max_idx = len(lq_frames_path)
            assert max_idx == len(lq_frames_path), (
                    f'Different number of images in lq ({max_idx})'
                    f' and gt folders ({len(gt_frames_path)})')

            self.data_info['lq_path'].extend(lq_frames_path)
            self.data_info['gt_path'].extend(gt_frames_path)
            self.data_info['clip_name'].append(clip_name)
            self.data_info['max_idx'].append(max_idx)
            
            self.lq_frames[clip_name] = lq_frames_path 
            self.gt_frames[clip_name] = gt_frames_path 
        
    def __getitem__(self, index):
        clip_name = self.data_info['clip_name'][index]
        max_idx = self.data_info['max_idx'][index]
        select_idx = range(int(max_idx))
        
        lq_frames_path = [self.lq_frames[clip_name][i] for i in select_idx]
        gt_frames_path = [self.gt_frames[clip_name][i] for i in select_idx]
        
        frame_list = list(range(len(lq_frames_path)))
        
        lq_frames = read_img_seq(lq_frames_path) 
        gt_frames = read_img_seq(gt_frames_path)
        
        return {
            'lq': lq_frames,  # (t, c, h, w)
            'gt': gt_frames,  # (t, c, h, w)
            'clip_name': clip_name,
            'frame_list': frame_list
        }

    def __len__(self):
        return len(self.data_info['clip_name'])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='REDS4')
    parser.add_argument('--dataroot_lq', type=str, default='/root/proj/Datasets/Vid4/BIx4')
    parser.add_argument('--dataroot_gt', type=str, default='/root/proj/Datasets/Vid4/GT')
    parser.add_argument('--cache_data', type=bool, default=False)
    opt = parser.parse_args()

    vid4_dataset = VidTestDataset(opt)
    dataloader_args = dict(dataset=vid4_dataset, batch_size=1, shuffle=False, num_workers=0)
    dataloader = data.DataLoader(**dataloader_args)

    for idx, val_data in enumerate(dataloader):
        print('idx:{0}, video:{1}, lq_data:{2}, gt_data:{3}'.format(idx, val_data['clip_name'], val_data['lq'].shape, val_data['gt'].shape))
