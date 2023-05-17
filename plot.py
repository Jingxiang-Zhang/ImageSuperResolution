import matplotlib.pyplot as plt

def show_img(img_hr, img_lr):
    plt.figure(figsize=(14,10))
    plt.subplot(1,2,1)
    plt.xticks([])
    plt.yticks([])
    plt.title("high resolution image")
    plt.imshow(img_hr)
    plt.subplot(1,2,2)
    plt.xticks([])
    plt.yticks([])
    plt.title("low resolution image")
    plt.imshow(img_lr)
    plt.show()

