import h5py
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchmetrics
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
from torchvision.transforms.functional import crop, pad
from tqdm import tqdm
import warnings
import math

warnings.filterwarnings("ignore")


def crop_image_HR_to_4_patches(lr_path, lr_save_path):
    """
    this function crop the high resolution image to 4 patches, padding
    """
    with tqdm(total=550) as t:
        for i in range(1, 551):
            lrp = os.path.join(lr_path, "{:0>4}".format(i) + "x2.png")
            lr = Image.open(lrp)
            W, H = lr.size
            if W == 1020 and H < 1020:
                padH = (510 - H % 510) // 2
                lr = pad(lr, padding=[0, padH])
            elif H == 1020 and W < 1020:
                padW = 510 - (W % 510) // 2
                lr = pad(lr, padding=[padW, 0])
            nW, nH = lr.size
            save_path = os.path.join(lr_save_path, str(i))
            bimg = crop(lr, 0, 0, 510, 510)
            bimg.save(save_path + "-1.png")
            if nW > 510:
                bimg = crop(lr, 0, 510, 510, 510)
                bimg.save(save_path + "-2.png")
            if nH > 510:
                bimg = crop(lr, 510, 0, 510, 510)
                bimg.save(save_path + "-3.png")
            if nW > 510 and nH > 510:
                bimg = crop(lr, 510, 510, 510, 510)
                bimg.save(save_path + "-4.png")
            t.update(1)


def crop_image_LR_to_4_patches(hr_path, hr_save_path):
    """
    this function crop the low resolution image to 4 patches, padding
    """
    with tqdm(total=550) as t:
        for i in range(1, 551):
            hrp = os.path.join(hr_path, "{:0>4}".format(i) + ".png")
            hr = Image.open(hrp)
            W, H = hr.size
            if W == 2040 and H < 2040:
                padH = (1020 - H % 1020) // 2
                hr = pad(hr, padding=[0, padH])
            elif H == 2040 and W < 2040:
                padW = 1020 - (W % 1020) // 2
                hr = pad(hr, padding=[padW, 0])
            nW, nH = hr.size
            save_path = os.path.join(hr_save_path, str(i))
            bimg = crop(hr, 0, 0, 1020, 1020)
            bimg.save(save_path + "-1.png")
            if nW > 1020:
                bimg = crop(hr, 0, 1020, 1020, 1020)
                bimg.save(save_path + "-2.png")
            if nH > 1020:
                bimg = crop(hr, 1020, 0, 1020, 1020)
                bimg.save(save_path + "-3.png")
            if nW > 1020 and nH > 1020:
                bimg = crop(hr, 1020, 1020, 1020, 1020)
                bimg.save(save_path + "-4.png")
            t.update(1)


def create_train_data_Y_channel(h5_train_path, hr_path, lr_path, interpolation=False):
    """
    create train data by the data set folder created by crop_image_LR_to_4_patches
    and crop_image_HR_to_4_patches
    """
    length_list = list()
    for number in range(6):
        image_number_range = range(number * 100 + 1, number * 100 + 101)
        h5path = os.path.join(h5_train_path, str(number) + ".hdf5")
        length = create_train_data_Y_channel_core(hr_path, lr_path, h5path, image_number_range)
        length_list.append(length)
    index_file = os.path.join(h5_train_path, "index.txt")
    with open(index_file, "w") as file:
        for length in length_list:
            file.write(str(length) + "\n")


def create_train_data_Y_channel_core(hr_path, lr_path, h5path, image_number_range, interpolation=False):
    """
    called by create_train_data_Y_channel, read 100 numbers of LR and HR images,
    save to hdf5 file
    """
    hrPatches = list()
    lrPatches = list()
    with tqdm(total=len(image_number_range)) as t:
        for image_number in image_number_range:
            hrp_base = os.path.join(hr_path, str(image_number))
            lrp_base = os.path.join(lr_path, str(image_number))
            for i in range(1, 5):
                hrp = hrp_base + "-" + str(i) + ".png"
                lrp = lrp_base + "-" + str(i) + ".png"
                if not os.path.exists(hrp):
                    continue
                hr = cv2.imread(hrp)
                hr = cv2.cvtColor(hr, cv2.COLOR_BGR2YCR_CB)  # for hr, change to ycbcr
                lr = cv2.imread(lrp)
                lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
                if interpolation:
                    lr = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)
                lr = cv2.cvtColor(lr, cv2.COLOR_RGB2YCR_CB)
                hr = np.array(hr).astype(np.float32)[:, :, 0] / 255  # get Y channel
                # hr = hr[padding:-padding,padding:-padding]
                lr = np.array(lr).astype(np.float32)[:, :, 0] / 255  # get Y channel
                hr = np.expand_dims(hr, axis=(0))
                lr = np.expand_dims(lr, axis=(0))
                hrPatches.append(hr)
                lrPatches.append(lr)
            t.update(1)
    length = len(hrPatches)
    h5_file = h5py.File(h5path, 'w')
    h5_file.create_dataset('lr', data=np.array(lrPatches))
    h5_file.create_dataset('hr', data=np.array(hrPatches))
    h5_file.close()
    return length

def validation_interpolation(validation_path, validation_interpolation_path):
    """
    interpolate validation set
    """
    dir_list = os.listdir(validation_path)
    with tqdm(total=len(dir_list)) as t:
        for dir_name in dir_list:
            path = os.path.join(validation_path, dir_name)
            lr = cv2.imread(path)
            lr = cv2.resize(lr, (lr.shape[1]*2, lr.shape[0]*2), interpolation=cv2.INTER_CUBIC)
            write_path = os.path.join(validation_interpolation_path, dir_name)
            cv2.imwrite(write_path, lr)
            t.update(1)