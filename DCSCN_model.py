import torch
import torch.nn as nn
import os
import numpy as np
from criteria import SSIM, PSNR
from tqdm import tqdm
import math
import cv2
from plot import show_img
from torch.nn import functional as F


# net reference: https://github.com/jmrf/dcscn-super-resolution/blob/master/dcscn/net.py

class DCSCN(nn.Module):
    def __init__(self, num_channels=1):
        super(DCSCN, self).__init__()
        self.in_channels = num_channels
        self._set_parameters()
        self._build_model()

    def _set_parameters(self):
        # training parameters
        self.dropout = 0.8
        self.weight_decay = 1e-4
        # feature extraction parameters
        self.conv_n_filters = [96, 81, 70, 60, 50, 41, 32]
        self.conv_kernels = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
        # reconstruction parameters
        self.scale_factor = 2
        self.reconstruction_n_filters = {'A1': 64, 'B1': 32, 'B2': 32}
        self.reconstruction_kernels = {'A1': (1, 1), 'B1': (1, 1), 'B2': (3, 3)}

    def _build_model(self):
        # feature extraction network
        in_channels = self.in_channels
        self.conv_sets = nn.ModuleList()
        for s_i, n_filters in enumerate(self.conv_n_filters):
            self.conv_sets.append(
                self._build_conv_set(in_channels, n_filters,
                                     kernel=self.conv_kernels[s_i]))
            in_channels = n_filters

        # reconstruction network
        in_channels = np.sum(self.conv_n_filters)
        self.reconstruction = nn.ModuleDict()
        # A1
        self.reconstruction['A1'] = self._build_reconstruction_conv(
            in_channels,
            self.reconstruction_n_filters['A1'],
            kernel=self.reconstruction_kernels['A1']
        )
        # B1
        self.reconstruction['B1'] = self._build_reconstruction_conv(
            in_channels,
            self.reconstruction_n_filters['B1'],
            kernel=self.reconstruction_kernels['B1']
        )
        # B2
        self.reconstruction['B2'] = self._build_reconstruction_conv(
            self.reconstruction_n_filters['B1'],
            self.reconstruction_n_filters['B2'],
            kernel=self.reconstruction_kernels['B2'],
            padding=1
        )
        # last convolution
        inp_channels = self.reconstruction_n_filters['B2'] + \
                       self.reconstruction_n_filters['A1']
        self.l_conv = nn.Conv2d(
            inp_channels,
            self.scale_factor ** 2,
            (1, 1)
        )
        self.PixelShuffle = nn.PixelShuffle(self.scale_factor)

    def _build_conv_set(self, in_channels, out_channels, kernel):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                       stride=1, padding=1,
                                       kernel_size=kernel),
                             nn.PReLU(),
                             nn.Dropout(p=self.dropout))

    def _build_reconstruction_conv(self, in_channels,
                                   out_channels, kernel, padding=0):
        return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels,
                                                padding=padding,
                                                kernel_size=kernel),
                             nn.PReLU(),
                             nn.Dropout(p=self.dropout))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight, a=0)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """Forward model pass.
        Arguments:
            x Tensor -- Batch of image tensors o shape: B x C x H x W
        Returns:
            Tensor -- Super Resolution upsampled image
        """
        # bi-cubic upsampling
        x_up = F.interpolate(x, mode='bicubic',
                             scale_factor=self.scale_factor)

        # convolutions and skip connections
        features = []
        for conv in self.conv_sets:
            x_new = conv.forward(x)
            features.append(x_new)
            x = x_new

        # concatenation 1: through filter dimensions
        x = torch.cat(features, 1)

        # reconstruction part
        a1_out = self.reconstruction['A1'].forward(x)
        b1_out = self.reconstruction['B1'].forward(x)
        b2_out = self.reconstruction['B2'].forward(b1_out)

        # concatenation 2 & last convolution
        x = torch.cat([a1_out, b2_out], 1)
        x = self.l_conv.forward(x)  # outputs a quad-image

        # network output + bicubic upsampling:
        # the 4 channels of the output represent the 4 corners
        # of the resulting image
        x = self.PixelShuffle(x) + x_up
        return x


def get_DCSCN_model(model_save_path=None):
    """
    load SRCNN model, if no model_save_path, then init a new model
    """
    model = DCSCN()
    if (model_save_path == None) or (not os.path.exists(model_save_path)):
        print("init new model parameter")
        model.init_weights()
        current_epoch = 0
        PSNR_list = list()
        SSIM_list = list()
    else:
        para = torch.load(model_save_path)
        model.load_state_dict(para["state_dict"])
        current_epoch = para["epoch"]
        PSNR_list = para["psnr"]
        SSIM_list = para["ssim"]
        print("load model parameter")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("use device: ", device)
    return model, device, current_epoch, PSNR_list, SSIM_list


def test_psnr_ssim(model, device, val_dataloader):
    """
    test PSNR and SSIM for each epoch
    """
    model.eval()
    psnr_list = list()
    ssim_list = list()
    padding = 6
    for hr, lr in val_dataloader:
        hr = hr.numpy() * 255  # read hr from data loader
        hr = np.squeeze(hr)  # change shape to a image
        hr = hr[:, :, 0]  # get Y channel

        lr_y = lr[:, :, :, 0]  # get Y channel
        # img is too large to put into this model
        _, w, h = lr_y.shape
        lr_y_part = [
            lr_y[:, :w // 2 + 10, :h // 2 + 10],
            lr_y[:, w // 2 - 10:, :h // 2 + 10],
            lr_y[:, :w // 2 + 10, h // 2 - 10:],
            lr_y[:, w // 2 - 10:, h // 2 - 10:]
        ]
        lr_y_collect = list()

        for lr_y_temp in lr_y_part:
            lr_y_temp = torch.reshape(lr_y_temp, (1, 1, lr_y_temp.shape[1], lr_y_temp.shape[2]))
            # reshape Y channel that fit the input
            lr_y_temp = lr_y_temp.to(device)  # put into model
            lr_y_temp = model(lr_y_temp)
            lr_y_temp = lr_y_temp.cpu().data.numpy()  # get the model result
            lr_y_temp = np.reshape(lr_y_temp, newshape=(lr_y_temp.shape[2], lr_y_temp.shape[3]))
            # reshape back to Y channel
            lr_y_temp[lr_y_temp > 1] = 1  # cut Y channel, 16<=Y<=235
            lr_y_temp[lr_y_temp < 0] = 0
            lr_y_temp = lr_y_temp * 255
            lr_y_collect.append(lr_y_temp)

        # combine together
        w = w * 2
        h = h * 2
        lr_y = np.zeros(shape=(w, h))
        lr_y[:w // 2, :h // 2] = lr_y_collect[0][:-20, :-20]
        lr_y[w // 2:, :h // 2] = lr_y_collect[1][20:, :-20]
        lr_y[:w // 2, h // 2:] = lr_y_collect[2][:-20, 20:]
        lr_y[w // 2:, h // 2:] = lr_y_collect[3][20:, 20:]

        psnr = PSNR(lr_y, hr, 255)  # test the result for RSCNN
        psnr_list.append(psnr)
        ssim = SSIM(lr_y, hr)
        ssim_list.append(ssim)
    return np.average(psnr_list), np.average(ssim_list)


def DCSCN_train(model, device, train_dataloader, val_dataloader,
                current_epoch, PSNR_list, SSIM_list, model_save_path,
                max_epoch=10):
    lr_begin = 0.001
    criterion = nn.MSELoss()

    with tqdm(total=len(train_dataloader) * max_epoch) as t:
        t.update(len(train_dataloader) * current_epoch)  # update to current state
        while current_epoch < max_epoch:
            lr = math.pow(0.90, current_epoch) * lr_begin
            optimizer = torch.optim.Adam(params=model.parameters(), weight_decay=1e-4, lr=0.01)
            model.train()
            for hr, lr in train_dataloader:
                hr, lr = hr.to(device), lr.to(device)
                lr_after = model(lr)
                loss = criterion(lr_after, hr)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t.update(1)
            # test result
            current_epoch += 1
            psnr, ssim = test_psnr_ssim(model, device, val_dataloader)
            PSNR_list.append(psnr)
            SSIM_list.append(ssim)
            torch.save({"epoch": current_epoch, "state_dict": model.state_dict(),
                        "psnr": PSNR_list, "ssim": SSIM_list}, model_save_path)


def test_one_picture_in_val_set(model, device, val_dataloader, save_dir,
                                img_number=0):
    for hr, lr in val_dataloader:
        if img_number != 0:
            img_number -= 1
            continue
        # get the hr image, and hr_y channel
        hr = hr.numpy() * 255  # read hr from data loader
        hr = np.reshape(hr, (hr.shape[1], hr.shape[2], hr.shape[3]))  # change shape to a image
        hr_y = hr[:, :, 0]  # get Y channel of hr
        hr = np.array(hr).astype(np.uint8)  # change to uint8
        hr = cv2.cvtColor(hr, cv2.COLOR_YCR_CB2RGB)  # change back to RGB

        # change lr to numpy at first, and interpolation to get the high resolution CbCr
        lr_np = lr.numpy() * 255  # read lr from data loader
        lr_np = np.squeeze(lr_np)  # change shape to a image
        lr_np = lr_np.astype(np.uint8)  # change to uint8
        lr_np = cv2.cvtColor(lr_np, cv2.COLOR_YCR_CB2RGB)  # convert to RGB first
        lr_np = cv2.resize(lr_np, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)  # interpolation
        lr_img = lr_np
        lr_np = cv2.cvtColor(lr_np, cv2.COLOR_RGB2YCR_CB)  # convert to YCbCr again
        lr_y = lr_np[:, :, 0]
        lr_Cb = lr_np[:, :, 1]
        lr_Cb = np.expand_dims(lr_Cb, axis=(0))
        lr_Cr = lr_np[:, :, 2]
        lr_Cr = np.expand_dims(lr_Cr, axis=(0))

        # Y channel will be got from model
        lr_y_after = lr[:, :, :, 0]  # get Y channel
        # Y is too large to put into the model, it need to be cutted
        _, w, h = lr_y_after.shape
        lr_y_after_part = [
            lr_y_after[:, :w // 2 + 10, :h // 2 + 10],
            lr_y_after[:, w // 2 - 10:, :h // 2 + 10],
            lr_y_after[:, :w // 2 + 10, h // 2 - 10:],
            lr_y_after[:, w // 2 - 10:, h // 2 - 10:]
        ]
        lr_y_after_collect = list()
        for lr_y_after_temp in lr_y_after_part:
            lr_y_after_temp = torch.reshape(lr_y_after_temp,
                                            (1, 1, lr_y_after_temp.shape[1], lr_y_after_temp.shape[2]))
            # reshape Y channel that fit the input
            lr_y_after_temp = lr_y_after_temp.to(device)  # put into model
            lr_y_after_temp = model(lr_y_after_temp)
            lr_y_after_temp = lr_y_after_temp.cpu().data.numpy()  # get the model result
            lr_y_after_temp = np.reshape(lr_y_after_temp,
                                         newshape=(lr_y_after_temp.shape[2], lr_y_after_temp.shape[3]))
            # reshape back to Y channel
            lr_y_after_temp[lr_y_after_temp > 1] = 1  # cut Y channel, 16<=Y<=235
            lr_y_after_temp[lr_y_after_temp < 0] = 0
            lr_y_after_temp = lr_y_after_temp * 255
            lr_y_after_collect.append(lr_y_after_temp)

        # combine together
        w = w * 2
        h = h * 2
        lr_y_after = np.zeros(shape=(w, h))
        lr_y_after[:w // 2, :h // 2] = lr_y_after_collect[0][:-20, :-20]
        lr_y_after[w // 2:, :h // 2] = lr_y_after_collect[1][20:, :-20]
        lr_y_after[:w // 2, h // 2:] = lr_y_after_collect[2][:-20, 20:]
        lr_y_after[w // 2:, h // 2:] = lr_y_after_collect[3][20:, 20:]
        lr_y_after = np.expand_dims(lr_y_after, axis=0)

        # combine together
        lr_after = np.concatenate([lr_y_after, lr_Cb, lr_Cr], axis=0)  # conbine the three channel together
        lr_after = np.transpose(lr_after, (1, 2, 0))  # to normal range
        lr_after = lr_after.astype(np.uint8)  # convert to uint8 type
        lr_after = cv2.cvtColor(lr_after, cv2.COLOR_YCR_CB2RGB)  # change back to RGB

        # demonstrate
        lr_y_after = np.squeeze(lr_y_after)
        show_img(hr, lr_after)
        psnr = PSNR(hr_y, lr_y, 255)  # test the result for previous interpolation
        print("Y channel PSNR value after interpolation: {:.3f}".format(psnr))
        psnr = PSNR(hr_y, lr_y_after, 255)  # test the result for RSCNN
        print("Y channel PSNR value after DCSCN: {:.3f}".format(psnr))
        ssim = SSIM(hr_y, lr_y)  # test the result for previous interpolation
        print("Y channel SSIM value after interpolation: {:.3f}".format(ssim))
        ssim = SSIM(hr_y, lr_y_after)  # test the result for RSCNN
        print("Y channel SSIM value after DCSCN: {:.3f}".format(ssim))

        # save image
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_RGB2BGR)
        hr = cv2.cvtColor(hr, cv2.COLOR_RGB2BGR)
        lr_after = cv2.cvtColor(lr_after, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(save_dir, "origin.png"), hr)
        cv2.imwrite(os.path.join(save_dir, "interpolation.png"), lr_img)
        cv2.imwrite(os.path.join(save_dir, "DCSCN.png"), lr_after)
        break
