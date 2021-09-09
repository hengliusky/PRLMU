import os
import sys
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

try:
    sys.path.append('..')
    from data.util import imresize
    import utils as util
except ImportError:
    pass


def generate_mod_LR_bic():
    # set parameters
    up_scale = 4
    mod_scale = 4
    # set data dir
    sourcedir = "/opt/data/us_data/us_test/us_images/"
    savedir = "/opt/data/us_data/us_test/us_images/"

    # load PCA matrix of enough kernel
    print("load PCA matrix")
    pca_matrix = torch.load(
        "../../pca_matrix.pth", map_location=lambda storage, loc: storage
    )
    print("PCA matrix shape: {}".format(pca_matrix.shape))

    degradation_setting = {
        "random_kernel": False,
        "code_length": 10,
        "ksize": 21,
        "pca_matrix": pca_matrix,
        "scale": up_scale,
        "cuda": True,
        "rate_iso": 1.0
    }

    # set random seed
    util.set_random_seed(0)

    filepaths = sorted([f for f in os.listdir(sourcedir) if f.endswith(".jpg")])
    print(filepaths)
    num_files = len(filepaths)

    # kernel_map_tensor = torch.zeros((num_files, 1, 10)) # each kernel map: 1*10

    # prepare data with augementation

    for i in range(num_files):
        filename = filepaths[i]
        print("No.{} -- Processing {}".format(i, filename))
        # read image
        image = cv2.imread(os.path.join(sourcedir, filename))

        width = int(np.floor(image.shape[1] // (mod_scale * mod_scale)))
        height = int(np.floor(image.shape[0] // (mod_scale * mod_scale)))
        print(width, height)
        # modcrop
        if len(image.shape) == 3:
            image_HR = image[0: mod_scale * height * mod_scale, 0: mod_scale * mod_scale * width, :]
        else:
            image_HR = image[0: mod_scale * mod_scale * height, 0: mod_scale * mod_scale * width]
        # LR_blur, by random gaussian kernel
        img_HR = util.img2tensor(image_HR)
        C, H, W = img_HR.size()

        for sig in np.linspace(1.8, 3.2, 8):
            prepro = util.SRMDPreprocessing(sig=sig, **degradation_setting)

            LR_img, ker_map, kernels = prepro(img_HR.view(1, C, H, W))
            print(kernels.shape)
            # cv2.imwrite(os.path.join('./kernels/', 'sig{}_{}'.format(sig, filename)), kernels)
            # plt.imsave('./kernels/sig{}_{}'.format(sig, filename), kernels)
            kernels = kernels.to('cpu').numpy()
            im = Image.fromarray(kernels)
            im.save('./kernels/sig{}_{}'.format(sig, filename))
            '''kernels = kernels.cpu().squeeze(0)

            plt.imshow(kernels, cmap='gray')
            plt.pause(1)
            kernels = np.array(kernels)
            print(type(kernels))
            sio.savemat(os.path.join('./kernels/','sig%.1f.mat' % sig), {'ker': kernels})'''
            '''image_LR_blur = util.tensor2img(LR_img)
            cv2.imwrite(os.path.join(saveLRblurpath, 'sig{}_{}'.format(sig,filename)), image_LR_blur)
            cv2.imwrite(os.path.join(saveHRpath, 'sig{}_{}'.format(sig,filename)), image_HR)
        # LR
        image_LR = imresize(image_HR, 1 / up_scale, True)
        # bic
        image_Bic = imresize(image_LR, up_scale, True)

        # cv2.imwrite(os.path.join(saveHRpath, filename), image_HR)
        cv2.imwrite(os.path.join(saveLRpath, filename), image_LR)
        cv2.imwrite(os.path.join(saveBicpath, filename), image_Bic)

        # kernel_map_tensor[i] = ker_map
    # save dataset corresponding kernel maps
    # torch.save(kernel_map_tensor, './Set5_sig2.6_kermap.pth')'''
    print("Image Blurring & Down smaple Done: X" + str(up_scale))


if __name__ == "__main__":
    generate_mod_LR_bic()