import numpy as np
import cv2
from plot import show_img
from tqdm import tqdm


def PSNR(img1, img2, max_num):
    """
    calculate PSNR for two images
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    else:
        return 20 * np.log10(max_num / np.sqrt(mse))


def SSIM(img1, img2):
    """
    calculate SSIM for two images
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    img1 = np.array(img1).astype(np.float64)
    img2 = np.array(img2).astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def show_img_in_loader_and_comparison(img_number, dataloader, hr_shrink=6):
    '''
    demonstrate one image and it corresponding label in dataloader, and calculate
    there SSIM and PSNR
    '''
    for hr, lr in dataloader:
        if img_number != 0:
            img_number -= 1
            continue
        hr = hr.numpy() * 255
        hr = np.reshape(hr, (hr.shape[1], hr.shape[2], hr.shape[3]))
        hr_y = hr[:, :, 0]
        hr = np.array(hr).astype(np.uint8)
        hr = cv2.cvtColor(hr, cv2.COLOR_YCR_CB2RGB)

        lr = lr.numpy() * 255
        lr = np.reshape(lr, (lr.shape[1], lr.shape[2], lr.shape[3]))
        lr = lr[hr_shrink:-hr_shrink, hr_shrink:-hr_shrink]
        lr_y = lr[:, :, 0]
        lr = np.array(lr).astype(np.uint8)
        lr = cv2.cvtColor(lr, cv2.COLOR_YCR_CB2RGB)
        show_img(hr, lr)
        psnr = PSNR(hr_y, lr_y, 255)
        print("Y channel PSNR value: {:.3f}".format(psnr))
        ssim = SSIM(hr_y, lr_y)
        print("Y channel SSIM value: {:.3f}".format(ssim))
        break


def test_psnr_ssim_interpolation(dataloader, hr_shrink=6):
    """
    make comparison for all image in validation loader
    """
    psnr_list = list()
    ssim_list = list()
    with tqdm(total=250) as t:
        for hr, lr in dataloader:
            hr = hr.numpy() * 255
            hr = np.squeeze(hr)
            hr = hr[:, :, 0]

            lr = lr.numpy() * 255
            lr = np.squeeze(lr)
            lr = lr[:, :, 0]
            if hr_shrink!=0:
                lr = lr[hr_shrink:-hr_shrink, hr_shrink:-hr_shrink]

            psnr = PSNR(hr, lr, 255)
            psnr_list.append(psnr)
            ssim = SSIM(hr, lr)
            ssim_list.append(ssim)
            t.update(1)

    return np.average(psnr_list), np.average(ssim_list)
