import torch
import torch.nn as nn
import os
import numpy as np
from criteria import SSIM, PSNR
from tqdm import tqdm
import math
import cv2
from plot import show_img


class ESPCN(nn.Module):
    def __init__(self, scale_factor, num_channels=1):
        super(ESPCN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.last_part = nn.Sequential(
            nn.Conv2d(32, num_channels * (scale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor)
        )
        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.in_channels == 32:
                    nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
                    nn.init.zeros_(m.bias.data)
                else:
                    nn.init.normal_(m.weight.data, mean=0.0,
                                    std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.last_part(x)
        return x


def get_ESPCN_model(model_save_path=None, channel=1):
    """
    load SRCNN model, if no model_save_path, then init a new model
    """
    model = ESPCN(2, channel)
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
    psnr_list = list()
    ssim_list = list()
    padding = 6
    for hr, lr in val_dataloader:
        hr = hr.numpy() * 255  # read hr from data loader
        hr = np.squeeze(hr)  # change shape to a image
        hr = hr[:, :, 0]  # get Y channel

        lr_y = lr[:, :, :, 0]  # get Y channel
        lr_y = torch.reshape(lr_y, (1, 1, lr.shape[1], lr.shape[2]))  # reshape Y channel that fit the input
        lr_y = lr_y.to(device)  # put into model
        lr_y = model(lr_y)
        lr_y = lr_y.cpu().data.numpy()  # get the model result
        lr_y = np.reshape(lr_y, newshape=(lr_y.shape[2], lr_y.shape[3]))  # reshape back to Y channel
        lr_y[lr_y > 1] = 1  # cut Y channel, 16<=Y<=235
        lr_y[lr_y < 0] = 0
        lr_y = lr_y * 255

        psnr = PSNR(lr_y, hr, 255)  # test the result for RSCNN
        psnr_list.append(psnr)
        ssim = SSIM(lr_y, hr)
        ssim_list.append(ssim)
    return np.average(psnr_list), np.average(ssim_list)


def ESPCN_train(model, device, train_dataloader, val_dataloader,
                current_epoch, PSNR_list, SSIM_list, model_save_path,
                max_epoch=10):
    lr_begin = 0.01
    criterion = nn.MSELoss()

    with tqdm(total=len(train_dataloader) * max_epoch) as t:
        t.update(len(train_dataloader) * current_epoch)  # update to current state
        while current_epoch < max_epoch:
            lr = math.pow(0.90, current_epoch) * lr_begin
            optimizer = torch.optim.Adam([
                {'params': model.first_part.parameters()},
                {'params': model.last_part.parameters(), 'lr': lr * 0.1}
            ], lr=lr)

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
        lr_y_after = torch.reshape(lr_y_after, (1, 1, lr.shape[1], lr.shape[2]))
        lr_y_after = lr_y_after.to(device)  # put into model
        lr_y_after = model(lr_y_after)
        lr_y_after = lr_y_after.cpu().data.numpy()  # get the model result
        lr_y_after = np.reshape(lr_y_after, newshape=(1, lr_y_after.shape[2], lr_y_after.shape[3]))
        lr_y_after[lr_y_after > 1] = 1  # cut Y channel, 16<=Y<=235
        lr_y_after[lr_y_after < 0] = 0
        lr_y_after = lr_y_after * 255

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
        print("Y channel PSNR value after ESPCN: {:.3f}".format(psnr))
        ssim = SSIM(hr_y, lr_y)  # test the result for previous interpolation
        print("Y channel SSIM value after interpolation: {:.3f}".format(ssim))
        ssim = SSIM(hr_y, lr_y_after)  # test the result for RSCNN
        print("Y channel SSIM value after ESPCN: {:.3f}".format(ssim))

        # save image
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_RGB2BGR)
        hr = cv2.cvtColor(hr, cv2.COLOR_RGB2BGR)
        lr_after = cv2.cvtColor(lr_after, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(save_dir, "origin.png"), hr)
        cv2.imwrite(os.path.join(save_dir, "interpolation.png"), lr_img)
        cv2.imwrite(os.path.join(save_dir, "ESPCN.png"), lr_after)
        break
