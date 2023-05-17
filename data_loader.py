import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import numpy as np
import h5py

class RSCNNTrainDataset(Dataset):
    def __init__(self, h5_train_path, hr_shrink):
        super(RSCNNTrainDataset, self).__init__()
        self.index = list()
        self.h5_train_path = h5_train_path
        self.hr_shrink = hr_shrink
        with open(os.path.join(h5_train_path, "index.txt")) as index_file:
            lines = index_file.readlines()
            for line in lines:
                self.index.append(int(line.strip()))

    def __getitem__(self, idx):
        for i in range(len(self.index)):
            if idx < self.index[i]:
                path = os.path.join(self.h5_train_path, str(i) + ".hdf5")
                with h5py.File(path, 'r') as f:
                    hr = f['hr'][idx]
                    if self.hr_shrink!=0:
                        hr = hr[:,self.hr_shrink:-self.hr_shrink, self.hr_shrink:-self.hr_shrink]
                    lr = f['lr'][idx]
                    return hr, lr
            else:
                idx -= self.index[i]

    def __len__(self):
        return np.sum(self.index)


class ValDataset(Dataset):
    def __init__(self, hr_shrink, valid_hr, valid_lr):
        super(ValDataset, self).__init__()
        self.hr_shrink = hr_shrink
        self.valid_hr = valid_hr
        self.valid_lr = valid_lr

    def __getitem__(self, idx):
        idx = idx + 551
        hrp = os.path.join(self.valid_hr, "{:0>4}".format(idx) + ".png")  # open the image of
        lrp = os.path.join(self.valid_lr, "{:0>4}".format(idx) + "x2.png")  # the val set
        hr = cv2.imread(hrp)
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2YCR_CB)  # for hr, change to ycbcr
        lr = cv2.imread(lrp)
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2YCR_CB)
        if self.hr_shrink!=0:
            hr = hr[self.hr_shrink:-self.hr_shrink, self.hr_shrink:-self.hr_shrink]
        
        hr = np.array(hr).astype(np.float32) / 255
        lr = np.array(lr).astype(np.float32) / 255
        return hr, lr

    def __len__(self):
        return 250


def create_train_val_data_loader(h5_train_path, valid_hr, valid_lr, hr_shrink=0, batch_size=1):
    train_dataset = RSCNNTrainDataset(h5_train_path, hr_shrink)
    ValDataset_ = ValDataset(hr_shrink, valid_hr, valid_lr)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  # pin_memory=True,
                                  drop_last=True
                                  )
    val_dataloader = DataLoader(dataset=ValDataset_,
                                batch_size=1,
                                )
    return train_dataloader, val_dataloader