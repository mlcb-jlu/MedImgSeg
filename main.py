from CBFNet import CBFNet
import argparse
from utils import *
import cv2
from libsvm import svmutil
"""parsing and configuration"""


def parse_args():
    desc = "Pytorch implementation of CBFNet"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='[train / test]')
    parser.add_argument('--light', type=str2bool, default=False, help='[CBFNet full version / CBFNet light version]')
    parser.add_argument('--dataset', type=str, default='YOUR_DATASET_NAME', help='dataset_name')
    parser.add_argument('--folder', type=str, default='YOUR_FOLDER_NAME', help='folder_name')

    parser.add_argument('--iteration', type=int, default=7000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=2, help='The size of batch size')
    parser.add_argument('--print_freq', type=int, default=50, help='The number of image print freq')
    parser.add_argument('--save_freq', type=int, default=50, help='The number of model save freq')
    parser.add_argument('--decay_flag', type=str2bool, default=False, help='The decay_flag')

    # parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.00002, help='The weight decay')
    parser.add_argument('--adv_weight', type=float, default=1, help='Weight for GAN')
    parser.add_argument('--cycle_weight', type=float, default=10, help='Weight for Cycle')
    # parser.add_argument('--identity_weight', type=int, default=10, help='Weight for Identity')
    parser.add_argument('--identity_weight', type=float, default=10, help='Weight for Identity')
    parser.add_argument('--cam_weight', type=int, default=1000, help='Weight for CAM')
    parser.add_argument('--recon_weight', type=int, default=10, help='Weight for reconstruction')
    # parser.add_argument('--cam_weight', type=int, default=4000, help='Weight for CAM')

    parser.add_argument('--ch', type=int, default=54, help='base channel number per layer')
    parser.add_argument('--n_res', type=int, default=4, help='The number of resblock')
    parser.add_argument('--n_dis', type=int, default=6, help='The number of discriminator layer')

    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')

    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the results')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Set gpu mode; [cpu, cuda]')
    parser.add_argument('--benchmark_flag', type=str2bool, default=False)
    parser.add_argument('--resume', type=str2bool, default=False)
    parser.add_argument('--stage', type=int, default=1, help='control the stage of training')

    return check_args(parser.parse_args())


"""checking arguments"""
def check_args(args):
    # --result_dir
    check_folder(os.path.join(args.result_dir, args.dataset, args.folder, 'model'+str(args.stage)))
    check_folder(os.path.join(args.result_dir, args.dataset, args.folder, 'img'+str(args.stage)))
    check_folder(os.path.join(args.result_dir, args.dataset, args.folder, 'test'+str(args.stage)))
    check_folder(os.path.join(args.result_dir, args.dataset, args.folder, 'results_mask'+str(args.stage)))
    check_folder(os.path.join(args.result_dir, args.dataset, args.folder, 'results_mask_reverse'+str(args.stage)))
    check_folder(os.path.join(args.result_dir, args.dataset, args.folder, 'iccv_labels'+str(args.stage)))
    check_folder(os.path.join(args.result_dir, args.dataset, args.folder, 'trainA_labels'+str(args.stage)))
    check_folder(os.path.join(args.result_dir, args.dataset, args.folder, 'crf_deal'+str(args.stage)))
    check_folder(os.path.join(args.result_dir, args.dataset, args.folder, "crf_deal_pseud_mask"+str(args.stage)))
    check_folder(os.path.join(args.result_dir, args.dataset, args.folder, 'pseud_results_mask_reverse'+str(args.stage)))
    check_folder(os.path.join(args.result_dir, args.dataset, args.folder, 'pseud_iccv_labels'+str(args.stage)))
    check_folder(os.path.join(args.result_dir, args.dataset, args.folder, 'pseud_results_mask'+str(args.stage)))
    check_folder(os.path.join(args.result_dir, args.dataset, args.folder, 'heatmap_A2B'+str(args.stage)))
    check_folder(os.path.join(args.result_dir, args.dataset, args.folder, 'fake_A2B'+str(args.stage)))
    check_folder(os.path.join(args.result_dir, args.dataset, args.folder, 'fake_A2B_2'+str(args.stage)))

    check_folder(os.path.join(args.result_dir, args.dataset, args.folder, 'results_mask' + str(args.stage)+'val'))
    check_folder(os.path.join(args.result_dir, args.dataset, args.folder, 'results_mask_reverse' + str(args.stage)+'val'))
    check_folder(os.path.join(args.result_dir, args.dataset, args.folder, 'iccv_labels' + str(args.stage)+'val'))
    check_folder(os.path.join(args.result_dir, args.dataset, args.folder, 'trainA_labels' + str(args.stage)+'val'))

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args


"""main"""
def main():
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    gan = CBFNet(args)

    # build graph
    gan.build_model()

    if args.phase == 'train':
        gan.train()
        print(" [*] Training finished!")

    if args.phase == 'test':
        gan.test()
        print(" [*] Test finished!")

    if args.phase == 'pseud_mask':
        gan.pseud_mask()
        print(" All pseud mask produced!")

    if args.phase == 'val':
        gan.val()
        print(" [*] Val finished!")


if __name__ == '__main__':
    main()
