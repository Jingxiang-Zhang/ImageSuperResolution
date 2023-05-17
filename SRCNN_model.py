import torch
import torch.nn as nn
import os
import numpy as np
from criteria import SSIM, PSNR
from tqdm import tqdm
import math
import cv2
from plot import show_img


class SRCNN(nn.Module):
    """
    SRCNN modle
    """

    def __init__(self, padding=False, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(num_channels, 64, kernel_size=9, padding=4 * int(padding),
                                             padding_mode='replicate'),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1, padding=0),  # n1 * 1 * 1 * n2
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2 * int(padding), padding_mode='replicate')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def init_weights(self):
        for L in self.conv1:
            if isinstance(L, nn.Conv2d):
                L.weight.data.normal_(mean=0.0, std=0.001)
                L.bias.data.zero_()
        for L in self.conv2:
            if isinstance(L, nn.Conv2d):
                L.weight.data.normal_(mean=0.0, std=0.001)
                L.bias.data.zero_()
        self.conv3.weight.data.normal_(mean=0.0, std=0.001)
        self.conv3.bias.data.zero_()


def get_SRCNN_model(model_save_path=None):
    """
    load SRCNN model, if no model_save_path, then init a new model
    """
    model = SRCNN(padding=True)
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


def SSNR_train(model, device, train_dataloader, val_dataloader,
               current_epoch, PSNR_list, SSIM_list, model_save_path,
               max_epoch=10):
    lr_begin = 0.01
    criterion = nn.MSELoss()

    with tqdm(total=len(train_dataloader) * max_epoch) as t:
        t.update(len(train_dataloader) * current_epoch)  # update to current state
        while current_epoch < max_epoch:
            lr = math.pow(0.95, current_epoch) * lr_begin
            optimizer = torch.optim.SGD([
                {'params': model.conv1.parameters()},
                {'params': model.conv2.parameters()},
                {'params': model.conv3.parameters(), 'lr': lr * 0.1}
            ], lr=lr, momentum=0.9)

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
                                img_number=0, padding=6):
    for hr, lr in val_dataloader:
        if img_number != 0:
            img_number -= 1
            continue
        hr = hr.numpy() * 255  # read hr from data loader
        hr = np.reshape(hr, (hr.shape[1], hr.shape[2], hr.shape[3]))  # change shape to a image
        hr_y = hr[:, :, 0]  # get Y channel of hr
        hr = np.array(hr).astype(np.uint8)  # change to uint8
        hr = cv2.cvtColor(hr, cv2.COLOR_YCR_CB2RGB)  # change back to RGB

        lr_y = lr[:, :, :, 0]  # get Y channel
        # reshape Y channel that fit the input
        lr_y = torch.reshape(lr_y, (1, 1, lr.shape[1], lr.shape[2]))
        lr_Cb = lr[:, :, :, 1].data.numpy()  # get Cb channel
        lr_Cb = lr_Cb[:, padding:-padding, padding:-padding]  # cut the boarder
        lr_Cr = lr[:, :, :, 2].data.numpy()  # get Cr channel
        lr_Cr = lr_Cr[:, padding:-padding, padding:-padding]  # cut the boarder

        lr_y = lr_y.to(device)  # put into model
        lr_y = model(lr_y)
        lr_y = lr_y.cpu().data.numpy()  # get the model result
        lr_y = np.reshape(lr_y, newshape=(1, lr_y.shape[2], lr_y.shape[3]))  # reshape back to Y channel
        lr_y[lr_y > 1] = 1  # cut Y channel, 16<=Y<=235
        lr_y[lr_y < 0] = 0

        lr_after_y = np.squeeze(lr_y) * 255
        lr_after = np.concatenate([lr_y, lr_Cb, lr_Cr], axis=0)  # conbine the three channel together
        lr_after = np.transpose(lr_after, (1, 2, 0)) * 255  # to normal range
        lr_after = lr_after.astype(np.uint8)  # convert to uint8 type
        lr_after = cv2.cvtColor(lr_after, cv2.COLOR_YCR_CB2RGB)  # change back to RGB

        lr = lr.numpy() * 255  # read lr from data loader
        lr = np.reshape(lr, (lr.shape[1], lr.shape[2], lr.shape[3]))  # change shape to a image
        lr_y = lr[:, :, 0]  # get Y channel
        lr_y = lr_y[padding:-padding, padding:-padding]
        lr = np.array(lr).astype(np.uint8)  # change to uint8
        lr = cv2.cvtColor(lr, cv2.COLOR_YCR_CB2RGB)  # change back to RGB
        lr = lr[padding:-padding, padding:-padding]

        show_img(hr, lr_after)
        psnr = PSNR(hr_y, lr_y, 255)  # test the result for previous interpolation
        print("Y channel PSNR value after interpolation: {:.3f}".format(psnr))
        psnr = PSNR(hr_y, lr_after_y, 255)  # test the result for RSCNN
        print("Y channel PSNR value after RSCNN: {:.3f}".format(psnr))
        ssim = SSIM(hr_y, lr_y)  # test the result for previous interpolation
        print("Y channel SSIM value after interpolation: {:.3f}".format(ssim))
        ssim = SSIM(hr_y, lr_after_y)  # test the result for RSCNN
        print("Y channel SSIM value after RSCNN: {:.3f}".format(ssim))

        # save image
        lr = cv2.cvtColor(lr, cv2.COLOR_RGB2BGR)
        hr = cv2.cvtColor(hr, cv2.COLOR_RGB2BGR)
        lr_after = cv2.cvtColor(lr_after, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(save_dir, "origin.png"), hr)
        cv2.imwrite(os.path.join(save_dir, "interpolation.png"), lr)
        cv2.imwrite(os.path.join(save_dir, "PSNR.png"), lr_after)
        break


def test_one_given_picture(model, device, img_path, save_path, upscale=3, padding=6):
    lr = cv2.imread(img_path)
    lr = cv2.resize(lr, (lr.shape[1] * upscale, lr.shape[0] * upscale), interpolation=cv2.INTER_CUBIC)
    lr = cv2.cvtColor(lr, cv2.COLOR_BGR2YCR_CB)
    lr = lr.astype(np.float32) / 255

    lr_y = lr[:, :, 0]  # get Y channel
    lr_Cb = lr[:, :, 1]
    lr_Cb = np.array([lr_Cb[padding:-padding, padding:-padding]])  # cut the boarder
    lr_Cr = lr[:, :, 2]
    lr_Cr = np.array([lr_Cr[padding:-padding, padding:-padding]])  # cut the boarder

    w, h = lr_y.shape
    lr_y = torch.from_numpy(lr_y)
    lr_y = torch.reshape(lr_y, (1, 1, w, h))  # reshape Y channel that fit the input
    lr_y = lr_y.to(device)  # put into model
    lr_y = model(lr_y)
    lr_y = lr_y.cpu().data.numpy()  # get the model result
    lr_y = np.reshape(lr_y, newshape=(1, lr_y.shape[2], lr_y.shape[3]))  # reshape back to Y channel
    lr_y[lr_y > 1] = 1  # cut Y channel, 16<=Y<=235
    lr_y[lr_y < 0] = 0

    lr_after = np.concatenate([lr_y, lr_Cb, lr_Cr], axis=0)
    lr_after = np.transpose(lr_after, (1, 2, 0)) * 255  # to normal range

    lr_after = lr_after.astype(np.uint8)  # convert to uint8 type
    lr_after = cv2.cvtColor(lr_after, cv2.COLOR_YCR_CB2BGR)  # change back to RGB
    cv2.imwrite(save_path, lr_after)
