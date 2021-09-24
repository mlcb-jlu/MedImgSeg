
import numpy as np
import scipy.misc
import os.path
import cv2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import argparse
import imageio
from five_metrics_crf_deal import five_m

def crf_inference(img, probs, t=10, scale_factor=1, labels=21):


    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=3/scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=80/scale_factor, srgb=13, rgbim=np.copy(img), compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))

def _crf_with_alpha(cam_dict, alpha):
    v = np.array(list(cam_dict.values()))
    bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
    bgcam_score = np.concatenate((bg_score, v), axis=0)
    crf_score = crf_inference(orig_img, bgcam_score, labels=bgcam_score.shape[0])

    n_crf_al = dict()

    n_crf_al[0] = crf_score[0]
    for i, key in enumerate(cam_dict.keys()):
        n_crf_al[key+1] = crf_score[i+1]

    return n_crf_al

def parse_args():
    desc = "Pytorch implementation of U-GAT-IT"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--dataset', type=str, default='YOUR_DATASET_NAME', help='dataset_name')
    parser.add_argument('--folder', type=str, default='YOUR_FOLDER_NAME', help='folder_name')
    parser.add_argument('--max', type=str, default='YOUR_MAX_PTH_NAME', help='folder_name')
    parser.add_argument('--phase', type=str, default='test or val', help='folder_name')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    if(args.phase=='test'):
        stage = '1'


        # Output path of the result of dense CRF post-processing
        # out_cam_pred_path = os.path.join(root_path, "results", args.dataset, args.folder, "crf_deal1/")
        out_cam_pred_path = os.path.join("results", args.dataset, args.folder, "crf_deal1/")

        # orig_img_path = os.path.join(root_path, "dataset", args.dataset, args.folder, "testA", "images/")
        orig_img_path = os.path.join("dataset", args.dataset, args.folder, "testA", "images/")

        # max_txt = os.path.join(root_path, "results", args.dataset, args.folder, "max.txt")
        max_txt = os.path.join("results", args.dataset, args.folder, "max.txt")
        with open(max_txt, "r") as f:
            lines = f.readlines()
            max_pth = lines[-1].strip('\n')

        #
        # pre_path = os.path.join(root_path, "results", args.dataset, args.folder, "results_mask_reverse1", max_pth+"/")
        pre_path = os.path.join("results", args.dataset, args.folder, "results_mask_reverse1", max_pth+"/")

        # model_path = os.path.join(root_path, "results", args.dataset, args.folder, "model1/")
        model_path = os.path.join("results", args.dataset, args.folder, "model1/")
        model_list = os.listdir(model_path)
        for model in model_list:
            if(model!=max_pth):
                os.remove(model_path+model)


        pre_list = os.listdir(orig_img_path)

        for filename in pre_list:

            image_name = filename.split(".png")[0]
            print(image_name)
            image_path = os.path.join(orig_img_path, filename)
            orig_img = cv2.imread(image_path)
            orig_img_size = orig_img.shape
            index = filename.rfind(".")

            our_cam = cv2.imread(pre_path + image_name + "_a2_b.png", cv2.IMREAD_GRAYSCALE) /255
            cam_list=list()

            our_cam = np.reshape(our_cam,[1,256,256])
            our_cam = np.concatenate((our_cam,1 - our_cam),axis=0)
            cam_list.append(our_cam)
            sum_cam = np.sum(cam_list, axis=0)
            norm_cam = sum_cam / (np.max(sum_cam, (1, 2), keepdims=True) + 1e-5)

            cam_dict = {}

            for i in range(0,1):
                cam_dict[i] = norm_cam[i+1]

            if out_cam_pred_path is not None:
                bg_score = [np.ones_like(norm_cam[0])*0.5]
                pred = np.argmax(np.concatenate((bg_score, norm_cam)), 0)*255
                if not os.path.exists(out_cam_pred_path + "/CAM"):
                    os.makedirs(out_cam_pred_path + "/CAM")
                imageio.imsave(os.path.join(out_cam_pred_path+"/CAM", image_name + '.png'), norm_cam[1])


            bg_th = 40
            for t in [1,2,3,4,5,6,7,8]:
                    crf = _crf_with_alpha(cam_dict, t)

                    for i in range(256):
                        for j in range(256):
                            if(i+j<=2*bg_th):
                                crf[1][i][j] = 0
                            if (256-i + j <= 2 * bg_th):
                                crf[1][i][j] = 0
                            if (256-j + i <= 2 * bg_th):
                                crf[1][i][j] = 0
                            if (i + j <= 2 * bg_th):
                                crf[1][i][j] = 0
                            if (i + j >= 256*2-bg_th*2):
                                crf[1][i][j] = 0

                    if not os.path.exists(out_cam_pred_path+"/"+str(t)):
                        os.makedirs(out_cam_pred_path+"/"+str(t))
                    imageio.imsave(os.path.join(out_cam_pred_path+"/"+str(t), image_name + '.png'), crf[1] * 255)

        # label_path = os.path.join(root_path, "dataset", args.dataset, args.folder, "testA", "labels/")
        label_path = os.path.join("dataset", args.dataset, args.folder, "testA", "labels/")
        five_m(out_cam_pred_path, label_path)

    elif(args.phase=="val"):

        stage = '1'

        out_cam_pred_path = os.path.join("results", args.dataset, args.folder, 'val_folder', "crf_deal1/")

        orig_img_path = os.path.join("dataset", args.dataset, args.folder, "val", "images/")

        max_txt = os.path.join("results", args.dataset, args.folder, "val_max.txt")
        with open(max_txt, "r") as f:
            lines = f.readlines()
            max_pth = lines[-1].strip('\n')

        pre_path = os.path.join("results", args.dataset, args.folder, 'val_folder', "results_mask_reverse1", max_pth + "/")

        model_path = os.path.join("results", args.dataset, args.folder, "model1/")
        model_list = os.listdir(model_path)
        for model in model_list:
            if (model != max_pth):
                os.remove(model_path + model)

        pre_list = os.listdir(orig_img_path)

        for filename in pre_list:

            image_name = filename.split(".png")[0]
            print(image_name)
            image_path = os.path.join(orig_img_path, filename)
            orig_img = cv2.imread(image_path)
            orig_img_size = orig_img.shape

            index = filename.rfind(".")

            our_cam = cv2.imread(pre_path + image_name + "_a2_b.png", cv2.IMREAD_GRAYSCALE) / 255
            cam_list = list()

            our_cam = np.reshape(our_cam, [1, 256, 256])
            our_cam = np.concatenate((our_cam, 1 - our_cam), axis=0)
            cam_list.append(our_cam)

            sum_cam = np.sum(cam_list, axis=0)
            norm_cam = sum_cam / (np.max(sum_cam, (1, 2), keepdims=True) + 1e-5)


            cam_dict = {}
            # for i in range(20):
            for i in range(0, 1):
                cam_dict[i] = norm_cam[i + 1]


            if out_cam_pred_path is not None:
                bg_score = [np.ones_like(norm_cam[0]) * 0.5]
                pred = np.argmax(np.concatenate((bg_score, norm_cam)), 0) * 255
                if not os.path.exists(out_cam_pred_path + "/CAM"):
                    os.makedirs(out_cam_pred_path + "/CAM")
                imageio.imsave(os.path.join(out_cam_pred_path + "/CAM", image_name + '.png'), norm_cam[1])

            bg_th = 40
            for t in [1, 2, 3, 4, 5, 6, 7, 8]:
                crf = _crf_with_alpha(cam_dict, t)

                for i in range(256):
                    for j in range(256):
                        if (i + j <= 2 * bg_th):
                            crf[1][i][j] = 0
                        if (256 - i + j <= 2 * bg_th):
                            crf[1][i][j] = 0
                        if (256 - j + i <= 2 * bg_th):
                            crf[1][i][j] = 0
                        if (i + j <= 2 * bg_th):
                            crf[1][i][j] = 0
                        if (i + j >= 256 * 2 - bg_th * 2):
                            crf[1][i][j] = 0

                # crf[1] = 1 - crf[0]
                # crf[2] = 1 - crf[0]
                # scipy.misc.imsave('crf0.png', crf[0] * 255)
                # scipy.misc.imsave('crf1.png', crf[1] * 255)
                # scipy.misc.imsave('crf2.png', crf[2] * 255)
                # scipy.misc.imsave(os.path.join(args.out_cam_pred, img_name + '-crf0.png'), crf[0] * 255)
                # scipy.misc.imsave(os.path.join(args.out_cam_pred, img_name + '-crf1.png'), crf[1] * 255)
                # scipy.misc.imsave(os.path.join(args.out_cam_pred, img_name + '-crf2.png'), crf[2] * 255)
                if not os.path.exists(out_cam_pred_path + "/" + str(t)):
                    os.makedirs(out_cam_pred_path + "/" + str(t))
                imageio.imsave(os.path.join(out_cam_pred_path + "/" + str(t), image_name + '.png'), crf[1] * 255)

        label_path = os.path.join("dataset", args.dataset, args.folder, "val", "labels/")
        five_m(out_cam_pred_path, label_path)
    else:
        print("phase wrong")