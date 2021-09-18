import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def plt_hist(img):
    plt.hist(img.ravel(), 256, [0, 1])
    plt.show()
def iou_score(output, target):
    smooth = 1e-7
    # plt_hist(output)
    # plt_hist(target)
    output = output > 0.5  # 大于0.5为TRUE,小于0.5为FALSE
    target = target > 0.5

    intersection = (output & target).sum()
    union = (output | target).sum()

    TP = float(np.sum(np.logical_and(output == True, target == True)))
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = float(np.sum(np.logical_and(output == False, target == False)))
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = float(np.sum(np.logical_and(output == True, target == False)))
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = float(np.sum(np.logical_and(output == False, target == True)))

    #  calculate JA, Dice, SE, SP
    JA = TP / ((TP + FN + FP + 1e-7))
    AC = (TP + TN) / (TP + FP + TN + FN + 1e-7)
    DI = 2 * TP / (2 * TP + FN + FP + 1e-7)
    SE = TP / (TP + FN + 1e-7)
    SP = TN / ((TN + FP + 1e-7))

    return (intersection + smooth) / (union + smooth),JA,AC,DI,SE,SP

def five_m(folder_path, label_path):
    #pre
    # folder_path = r"/home/dw/Disk_8T/SY/zhourixin/github/UGATIT-2.0/results/BraTS/brats_1/crf_deal1/"
    # pre_path = r"/home/dw/Disk_8T/SY/pytorch-nested-unet-experiment2/大脑数据集743/labels"
    folder_list = os.listdir(folder_path)
    #label
    # label_path = r"/home/dw/Disk_8T/SY/pytorch-nested-unet-experiment2/大脑数据集743/GT"

    for folder in folder_list:
        count = 0
        iou_sum = 0
        JA_sum = 0
        AC_sum = 0
        DI_sum = 0
        SE_sum = 0
        SP_sum = 0
        pre_list = os.listdir(os.path.join(folder_path, folder))
        for filename in pre_list:
            count += 1
            image_path = os.path.join(folder_path, folder, filename)
            pre = cv2.imread(image_path)

            index = filename.rfind(".")
            label_name = filename[:index] + "_segmentation.png"
            target_path = os.path.join(label_path, label_name)
            target = cv2.imread(target_path)

            iou,JA,AC,DI,SE,SP = iou_score(pre/255, target/255)
            iou_sum += iou
            JA_sum += JA
            AC_sum += AC
            DI_sum += DI
            SE_sum += SE
            SP_sum += SP

        # print(iou_sum/count,JA_sum/count,AC_sum/count,DI_sum/count,SE_sum/count,SP_sum/count)
        print("folder:%s" % (folder))
        print("JA:%.4f " % (JA_sum/count))
        print("AC:%.4f " % (AC_sum/count))
        print("DI:%.4f " % (DI_sum/count))
        print("SE:%.4f " % (SE_sum/count))
        print("SP:%.4f " % (SP_sum/count))
        list_result = [JA_sum/count, AC_sum/count, DI_sum/count, SE_sum/count ,SP_sum/count]
        list_result_round = np.round(list_result, 4)
        print("  JA     AC     DI      SE     SP")
        print(", ".join(str(i) for i in list_result_round))
        if (float(list_result_round[0]) > 0.63 and float(list_result_round[0]) < 1):
            with open(folder_path + "results.txt", "a+") as f:
                f.write(folder + '\n')
                # f.write('\n')
                f.write(str(list_result))
                f.write('\n')
                f.write('==' * 5)
                f.write('\n')

            # print(round(count * 100 / len(pre_list), 2), "%")



