#!/usr/bin/env python
# coding=utf-8
from __future__ import division, print_function, unicode_literals, absolute_import
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from sacred import Ingredient

ds = Ingredient('dataset')


@ds.config
def cfg():
    name = 'balls4mass64'
    path = './data'
    train_size = None           # subset of training set (None, int)
    valid_size = 1000           # subset of valid set (None, int)
    test_size = None            # subset of test set (None, int)


ds.add_named_config('balls4mass64', {'name': 'balls4mass64'})
ds.add_named_config('balls678mass64', {'name': 'balls678mass64'})
ds.add_named_config('balls3curtain64', {'name': 'balls3curtain64'})
ds.add_named_config('atari', {'name': 'atari'})


class InputDataset(Dataset):
    @ds.capture
    def _open_dataset(self, out_list, path, name):
        # open dataset file
        self._hdf5_file = h5py.File(os.path.join(path, name + '.h5'), 'r')
        self._data_in_file = {
            data_name: self._hdf5_file[self.usage][data_name] for data_name in out_list
        }
      
        self.limit = self._data_in_file['features'].size()[1]

    def __init__(self, usage, out_list=('features', 'groups')):
        
        self.usage = usage
        
        # with tf.name_scope("{}_queue".format(usage[:5])):

        self._open_dataset(out_list)

    def __len__(self):
        return self.limit

    def __getitem__(self, index):
        data = [torch.from_numpy(ds[:self.sequence_length, index:index + 1][:, :, None]).cuda()
                     for data_name, ds in self._data_in_file.items()]
        return data
    

def collate(batch):
    data = [ [] for b in batch[0]]
    for b in batch:
        i=0
        for ten in b:
            data[i].append(b[i])
    data = [torch.stack(d,1) for d in data]
    return data

