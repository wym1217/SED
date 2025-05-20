from typing import List, Union
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import torchaudio
from torch.utils.data import Dataset
from functools import partial
import random

def load_dict_from_csv(file, cols, sep="\t"):
    if isinstance(file, str):
        df = pd.read_csv(file, sep=sep)
    elif isinstance(file, pd.DataFrame):
        df = file
    output = dict(zip(df[cols[0]], df[cols[1]]))
    return output


class InferenceDataset(Dataset):
    def __init__(self,
                 audio_file: Union[str, Path],
                 augment: bool = False,
                 augment_type: str = "time_shift"):
        super().__init__()
        self.aid_to_fpath = load_dict_from_csv(audio_file, ("audio_id", "file_name"))
        self.aids = list(self.aid_to_fpath.keys())
        self.type = augment_type
        self.augment = augment

    def __len__(self):
        return len(self.aids)

    def __getitem__(self, index):
        aid = self.aids[index]
        fpath = self.aid_to_fpath[aid]
        waveform, _ = torchaudio.load(fpath)
        if self.augment:
            waveform = self.augment_waveform(waveform)
        waveform = waveform.mean(0)
        waveform = torch.as_tensor(waveform).float()
        return aid, waveform
    # 数据增强
    def augment_waveform(self, waveform):
        length = len(waveform)
        if self.type == "time_shift":
            max_shift = max(1, int(length * 0.2))
            shift = random.choice([i for i in range(-max_shift, max_shift + 1) if i != 0])
            waveform = torch.roll(waveform, shifts=shift)
        elif self.type == "noise":
            noise = torch.randn_like(waveform) * 0.01
            waveform += noise
        elif self.type == "time_mask":
            begin = np.random.randint(0, waveform.shape[0])
            length = np.random.randint(0, 20)
            end = min(waveform.shape[0], begin + length)
            waveform[begin:end] = 0
        else:
            raise ValueError(f"Unknown augment type: {self.type}")
        return waveform


class TrainDataset(InferenceDataset):
    def __init__(self,
                 audio_file: Union[str, Path],
                 label_file: Union[str, Path],
                 label_to_idx: dict,
                 augment: bool = False,
                 augment_type: str = "time_shift"):
        super().__init__(audio_file, augment=augment, augment_type=augment_type)
        self.aid_to_label = load_dict_from_csv(label_file,
            ("filename", "event_labels"))
        self.aids = list(self.aid_to_label.keys())
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.aids)

    def __getitem__(self, index):
        aid, waveform = super().__getitem__(index)
        waveform = torch.as_tensor(waveform).float()
        label = self.aid_to_label[aid]
        target = torch.zeros(len(self.label_to_idx))
        for l in label.split(","):
            target[self.label_to_idx[l]] = 1
        return aid, waveform, target
    

def pad(tensorlist, batch_first=True, padding_value=0.):
    # In case we have 3d tensor in each element, squeeze the first dim (usually 1)
    if len(tensorlist[0].shape) == 3:
        tensorlist = [ten.squeeze() for ten in tensorlist]
    if isinstance(tensorlist[0], np.ndarray):
        tensorlist = [torch.as_tensor(arr) for arr in tensorlist]
    padded_seq = torch.nn.utils.rnn.pad_sequence(tensorlist,
                                                 batch_first=batch_first,
                                                 padding_value=padding_value)
    length = [tensor.shape[0] for tensor in tensorlist]
    return padded_seq, length


def sequential_collate_wrapper(batches, return_length=True, length_idxs: List=[]):
    seqs = []
    lens = []
    for idx, data_seq in enumerate(zip(*batches)):
        if isinstance(data_seq[0], (torch.Tensor, np.ndarray)):  # is tensor, then pad
            data_seq, data_len = pad(data_seq)
            if idx in length_idxs:
                lens.append(data_len)
        else:
            data_seq = np.array(data_seq)
        seqs.append(data_seq)
    if return_length:
        seqs.extend(lens)
    return seqs

def sequential_collate(return_length=True, length_idxs: List=[]):
    # 使用 partial 替代 lambda
    return partial(sequential_collate_wrapper, 
                  return_length=return_length, 
                  length_idxs=length_idxs)

