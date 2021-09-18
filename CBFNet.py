import time, itertools
from dataset import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from networks import *
from utils import *
from glob import glob
from vis_loss import Visualizer
from PIL import Image
from random import shuffle, randint
from findContours import findCont
from five_metrics import five_m
import sys

class CBFNet(object) :
    def __init__(self, args):
        self.light = args.light

        if self.light :
            self.model_name = 'CBFNet_light'
        else :
            self.model_name = 'CBFNet'

        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.folder = args.folder

        self.iteration = args.iteration
        self.decay_flag = args.decay_flag

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.cam_weight = args.cam_weight
        self.recon_weight = args.recon_weight

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.device = args.device
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume
        self.imglist_testA = os.listdir(os.path.join('dataset', self.dataset, self.folder, 'testA', 'images'))
        self.imglist_testB = os.listdir(os.path.join('dataset', self.dataset, 'testB'))
        self.batch_size = args.batch_size
        self.stage = args.stage

        if torch.backends.cudnn.enabled and self.benchmark_flag:
            print('set benchmark !')
            torch.backends.cudnn.benchmark = True

        print()

        print("##### Information #####")
        print("# light : ", self.light)
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch : ", self.iteration)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# identity_weight : ", self.identity_weight)
        print("# cam_weight : ", self.cam_weight)

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ DataLoader """
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Resize((self.img_size + 50, self.img_size+50)),
            transforms.RandomCrop(self.img_size),
            transforms.ColorJitter(contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        if(self.dataset=="ISIC"):
            dataset_A = "200_A_weak"
            dataset_B = "200_B_weak"
        elif(self.dataset=="BraTS"):
            dataset_A = "563_A_weak"
            dataset_B = "563_B_weak"
        else:
            print("dataset name is wrong")
            sys.exit(1)
        self.trainA = ImageFolder(os.path.join('dataset', self.dataset, self.folder, dataset_A), train_transform)
        self.trainB = ImageFolder(os.path.join('dataset', self.dataset, self.folder, dataset_B), train_transform)
        self.testA = ImageFolder(os.path.join('dataset', self.dataset, self.folder, 'testA', 'images'), test_transform)
        self.testB = ImageFolder(os.path.join('dataset', self.dataset, self.folder, 'testB'), test_transform)
        self.valA = ImageFolder(os.path.join('dataset', self.dataset, self.folder, 'val', 'images'), test_transform)
        self.pseud_A = ImageFolder(os.path.join('dataset', self.dataset, self.folder, dataset_A), test_transform)

        self.trainA_loader = DataLoader(self.trainA, batch_size=self.batch_size, shuffle=True)
        self.trainB_loader = DataLoader(self.trainB, batch_size=self.batch_size, shuffle=True)
        self.testA_loader = DataLoader(self.testA, batch_size=1, shuffle=False)
        self.testB_loader = DataLoader(self.testB, batch_size=1, shuffle=False)
        self.valA_loader = DataLoader(self.valA, batch_size=1, shuffle=False)
        self.pseud_A_loader = DataLoader(self.pseud_A, batch_size=1, shuffle=False)

        """ Define Generator, Discriminator """
        self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
        self.genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
        self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        self.disGB = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)
        self.disLB = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)

        """ Define Loss """
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)

        """ Trainer """
        p = itertools.chain(self.genA2B.parameters(), self.genB2A.parameters())
        self.G_optim = torch.optim.Adam(filter(lambda q : q.requires_grad, p), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)
        self.D_optim = torch.optim.Adam(itertools.chain(self.disGA.parameters(), self.disGB.parameters(), self.disLA.parameters(), self.disLB.parameters()), lr=self.lr, betas=(0.5, 0.999), weight_decay=self.weight_decay)

        """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
        self.Rho_clipper = RhoClipper(0, 1)

    def train(self):
        self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

        vis = Visualizer()

        start_iter = 1
        if self.resume:
            model_list = glob(os.path.join(self.result_dir, self.dataset, self.folder, 'model'+str(self.stage), '*.pt'))
            if not len(model_list) == 0:
                model_list.sort()
                start_iter = int(model_list[-1].split('_')[-1].split('.')[0])
                self.load(os.path.join(self.result_dir, self.dataset, self.folder, 'model'+str(self.stage)), start_iter)
                print(" [*] Load SUCCESS")
                if self.decay_flag and start_iter > (self.iteration // 2):
                    self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)
                    self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)

        # training loop
        print('training start !')
        start_time = time.time()
        imglist_testA = sorted(self.imglist_testA)
        imglist_testB = sorted(self.imglist_testB)
        test_start = 0

        for step in range(start_iter, self.iteration + 1):
            if self.decay_flag and step > (self.iteration // 2):
                self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))

            try:
                real_A, _, pathA = trainA_iter.next()
            except:
                trainA_iter = iter(self.trainA_loader)
                real_A, _, pathA = trainA_iter.next()

            try:
                real_B, _, pathB = trainB_iter.next()
            except:
                trainB_iter = iter(self.trainB_loader)
                real_B, _, pathB = trainB_iter.next()

            if(self.stage==2):
                imglist_trainA = os.listdir(os.path.join('dataset', self.dataset, 'syn_trainA'))
                shuffle(imglist_trainA)

                for t in range(self.batch_size):
                    # for img in imglist_trainA:
                    if(len(pathA)==self.batch_size):
                        img = str(pathA[t]).split('/')[-1]
                        tpath = os.path.join('dataset', self.dataset, 'syn_trainA', img.split('.png')[0]+'.png')  # 路径(/home/ouc/river/test)+图片名（img_m）
                        fopen = Image.open(tpath)
                        transform = transforms.Compose(
                            [transforms.ToTensor(),
                             transforms.Normalize(mean=(0.5), std=(0.5))])
                        real_A1 = transform(fopen)  # data就是预处理后，可以送入模型进行训练的数据
                        # real_A1 = torch.cat([real_A1,real_A1,real_A1],0)
                        real_A1 = real_A1.unsqueeze(0)
                        if(t==0):
                            realA_out = real_A1
                        else:
                            realA_out = torch.cat([realA_out, real_A1], 0)
                    else:
                        img = str(pathA[0]).split('/')[-1]
                        tpath = os.path.join('dataset', self.dataset, 'syn_trainA',
                                             img.split('.png')[0]+'.png')  # 路径(/home/ouc/river/test)+图片名（img_m）
                        fopen = Image.open(tpath)
                        transform = transforms.Compose(
                            [transforms.ToTensor(),
                             transforms.Normalize(mean=(0.5), std=(0.5))])
                        real_A1 = transform(fopen)  # data就是预处理后，可以送入模型进行训练的数据了
                        realA_out = real_A1.unsqueeze(0)
                        break
            else:
                realA_out = real_A

            # real_A2 = torch.cat([real_A, realA_out], 1)
            real_A2 = real_A

            realB_out = real_B
            # real_B2 = torch.cat([real_B, realB_out], 1)
            real_B2 = real_B

            real_A, real_B = real_A.to(self.device), real_B.to(self.device)
            real_A2, real_B2 = real_A2.to(self.device), real_B2.to(self.device)

            # adv_weight = self.adv_weight

            # Update D
            self.D_optim.zero_grad()

            fake_A2B, _, _, _, _, _ = self.genA2B(real_A2)
            fake_B2A, _, _, _, _, _ = self.genB2A(real_B2)

            real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
            real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
            real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
            real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
            fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
            fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

            D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)) + self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device))
            D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit, torch.ones_like(real_GA_cam_logit).to(self.device)) + self.MSE_loss(fake_GA_cam_logit, torch.zeros_like(fake_GA_cam_logit).to(self.device))
            D_ad_loss_LA = self.MSE_loss(real_LA_logit, torch.ones_like(real_LA_logit).to(self.device)) + self.MSE_loss(fake_LA_logit, torch.zeros_like(fake_LA_logit).to(self.device))
            D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit, torch.ones_like(real_LA_cam_logit).to(self.device)) + self.MSE_loss(fake_LA_cam_logit, torch.zeros_like(fake_LA_cam_logit).to(self.device))
            D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(self.device)) + self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))
            D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit, torch.ones_like(real_GB_cam_logit).to(self.device)) + self.MSE_loss(fake_GB_cam_logit, torch.zeros_like(fake_GB_cam_logit).to(self.device))
            D_ad_loss_LB = self.MSE_loss(real_LB_logit, torch.ones_like(real_LB_logit).to(self.device)) + self.MSE_loss(fake_LB_logit, torch.zeros_like(fake_LB_logit).to(self.device))
            D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit, torch.ones_like(real_LB_cam_logit).to(self.device)) + self.MSE_loss(fake_LB_cam_logit, torch.zeros_like(fake_LB_cam_logit).to(self.device))

            D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
            D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)

            Discriminator_loss = 1 * (D_loss_A + D_loss_B)
            Discriminator_loss.backward()
            self.D_optim.step()

            # Update G
            self.G_optim.zero_grad()

            fake_A2B, fake_A2B_cam_logit, fake_A2B_heatmap, att_maskA2B, contentA2B, realA_r = self.genA2B(real_A2)
            fake_B2A, fake_B2A_cam_logit, fake_B2A_heatmap, att_maskB2A, contentB2A, realB_r = self.genB2A(real_B2)

            fake_B2A2 = fake_B2A
            fake_B2A2B, _, _, _, _, _ = self.genA2B(fake_B2A2)

            fake_A2B2 = fake_A2B
            fake_A2B2A, _, _, _, _, _ = self.genB2A(fake_A2B2)

            fake_B2B, fake_B2B_cam_logit, _, _, content_B2B, _ = self.genA2B(real_B2)
            fake_A2A, fake_A2A_cam_logit, _, _, _, _ = self.genB2A(real_A2)

            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
            fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
            fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

            fake_GA_logit_f, fake_GA_cam_logit_f, _ = self.disGA(contentB2A)
            fake_LA_logit_f, fake_LA_cam_logit_f, _ = self.disLA(contentB2A)
            fake_GB_logit_f, fake_GB_cam_logit_f, _ = self.disGB(contentA2B)
            fake_LB_logit_f, fake_LB_cam_logit_f, _ = self.disLB(contentA2B)



            G_ad_loss_GA = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
            G_ad_cam_loss_GA = self.MSE_loss(fake_GA_cam_logit, torch.ones_like(fake_GA_cam_logit).to(self.device))
            G_ad_loss_LA = self.MSE_loss(fake_LA_logit, torch.ones_like(fake_LA_logit).to(self.device))
            G_ad_cam_loss_LA = self.MSE_loss(fake_LA_cam_logit, torch.ones_like(fake_LA_cam_logit).to(self.device))
            G_ad_loss_GB = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))
            G_ad_cam_loss_GB = self.MSE_loss(fake_GB_cam_logit, torch.ones_like(fake_GB_cam_logit).to(self.device))
            G_ad_loss_LB = self.MSE_loss(fake_LB_logit, torch.ones_like(fake_LB_logit).to(self.device))
            G_ad_cam_loss_LB = self.MSE_loss(fake_LB_cam_logit, torch.ones_like(fake_LB_cam_logit).to(self.device))
            adv_loss_A  = G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA
            adv_loss_B = G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB



            G_ad_loss_GA_f = self.MSE_loss(fake_GA_logit_f, torch.ones_like(fake_GA_logit_f).to(self.device))
            G_ad_cam_loss_GA_f = self.MSE_loss(fake_GA_cam_logit_f, torch.ones_like(fake_GA_cam_logit_f).to(self.device))
            G_ad_loss_LA_f = self.MSE_loss(fake_LA_logit_f, torch.ones_like(fake_LA_logit_f).to(self.device))
            G_ad_cam_loss_LA_f = self.MSE_loss(fake_LA_cam_logit_f, torch.ones_like(fake_LA_cam_logit_f).to(self.device))
            G_ad_loss_GB_f = self.MSE_loss(fake_GB_logit_f, torch.ones_like(fake_GB_logit_f).to(self.device))
            G_ad_cam_loss_GB_f = self.MSE_loss(fake_GB_cam_logit_f, torch.ones_like(fake_GB_cam_logit_f).to(self.device))
            G_ad_loss_LB_f = self.MSE_loss(fake_LB_logit_f, torch.ones_like(fake_LB_logit_f).to(self.device))
            G_ad_cam_loss_LB_f = self.MSE_loss(fake_LB_cam_logit_f, torch.ones_like(fake_LB_cam_logit_f).to(self.device))
            adv_loss_A_f = G_ad_loss_GA_f + G_ad_cam_loss_GA_f + G_ad_loss_LA_f + G_ad_cam_loss_LA_f
            adv_loss_B_f = G_ad_loss_GB_f + G_ad_cam_loss_GB_f + G_ad_loss_LB_f + G_ad_cam_loss_LB_f

            G_r_loss_A = self.L1_loss(realA_r, real_A)
            G_r_loss_B = self.L1_loss(realB_r, real_B)

            G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
            G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)


            G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
            G_identity_loss_B = self.L1_loss(fake_B2B, real_B)


            G_cam_loss_A = self.BCE_loss(fake_B2A_cam_logit, torch.ones_like(fake_B2A_cam_logit).to(self.device)) + self.BCE_loss(fake_A2A_cam_logit, torch.zeros_like(fake_A2A_cam_logit).to(self.device))
            G_cam_loss_B = self.BCE_loss(fake_A2B_cam_logit, torch.ones_like(fake_A2B_cam_logit).to(self.device)) + self.BCE_loss(fake_B2B_cam_logit, torch.zeros_like(fake_B2B_cam_logit).to(self.device))

            G_loss_A = self.adv_weight * (adv_loss_A+adv_loss_A_f) + self.cycle_weight * G_recon_loss_A + \
                       self.identity_weight * G_identity_loss_A + self.cam_weight * G_cam_loss_A + self.recon_weight*G_r_loss_A
            G_loss_B = self.adv_weight * (adv_loss_B+adv_loss_B_f) + self.cycle_weight * G_recon_loss_B + \
                       self.identity_weight * G_identity_loss_B + self.cam_weight * G_cam_loss_B + self.recon_weight*G_r_loss_B
            Generator_loss = G_loss_A + G_loss_B
            Generator_loss.backward()
            self.G_optim.step()

            # clip parameter of AdaILN and ILN, applied after optimizer step
            self.genA2B.apply(self.Rho_clipper)
            self.genB2A.apply(self.Rho_clipper)

            print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f, adv_loss: %.8f, identity_loss: %.8f, cam_loss: %.8f" %
                  (step, self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss,self.adv_weight * (adv_loss_A+adv_loss_B),
                self.identity_weight * (G_identity_loss_A+G_identity_loss_B), self.cam_weight * (G_cam_loss_A+G_cam_loss_B)))

            vis.plot_many_stack(step, Discriminator_loss, Generator_loss, self.adv_weight * (adv_loss_A+adv_loss_B),
                                self.cycle_weight*(G_recon_loss_A+G_recon_loss_B), self.identity_weight*(G_identity_loss_A+G_identity_loss_B), self.cam_weight*(G_cam_loss_A+G_cam_loss_B))


            if step % self.print_freq == 0:
                train_sample_num = 10
                # test_sample_num = 5
                A2B = np.zeros((self.img_size * 6, 0, 3))
                # B2A = np.zeros((self.img_size * 7, 0, 3))

                self.genA2B.eval(), self.genB2A.eval(), self.disGA.eval(), self.disGB.eval(), self.disLA.eval(), self.disLB.eval()
                for _ in range(train_sample_num):


                    imgA = imglist_testA[test_start % len(imglist_testA)]
                    tpath = os.path.join('dataset', self.dataset, self.folder, 'testA', 'images', imgA)
                    fopen = Image.open(tpath)
                    transform = transforms.Compose(
                        [transforms.ToTensor(),
                         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
                    real_A = transform(fopen)  # data就是预处理后，可以送入模型进行训练的数据了
                    real_A = real_A.unsqueeze(0)
                    real_A2 = real_A

                    imgB = imglist_testB[test_start % len(imglist_testB)]
                    tpath = os.path.join('dataset', self.dataset, 'testB', imgB)
                    fopen = Image.open(tpath)
                    transform = transforms.Compose(
                        [transforms.ToTensor(),
                         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
                    real_B = transform(fopen)  # data就是预处理后，可以送入模型进行训练的数据了
                    real_B = real_B.unsqueeze(0)
                    real_B2 = real_B

                    real_A, real_B = real_A.to(self.device), real_B.to(self.device)
                    real_A2, real_B2 = real_A2.to(self.device), real_B2.to(self.device)
                    test_start += 1

                    fake_A2B, _, fake_A2B_heatmap, att_maskA2B, contentA2B, real_A2_r = self.genA2B(real_A2)
                    fake_B2A, _, fake_B2A_heatmap, att_maskB2A, contentB2A, real_B2_r = self.genB2A(real_B2)

                    A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                               cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(att_maskA2B[0]))*2-1),
                                                               RGB2BGR(tensor2numpy(denorm(contentA2B[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(real_A2_r[0])))), 0)), 1)

                cv2.imwrite(os.path.join(self.result_dir, self.dataset, self.folder, 'img'+str(self.stage), 'A2B_%07d.png' % step), A2B * 255.0)
                self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

            if step % self.save_freq == 0:
                self.save(os.path.join(self.result_dir, self.dataset, self.folder, 'model'+str(self.stage)), step)

            if step % 1000 == 0:
                params = {}
                params['genA2B'] = self.genA2B.state_dict()
                params['genB2A'] = self.genB2A.state_dict()
                params['disGA'] = self.disGA.state_dict()
                params['disGB'] = self.disGB.state_dict()
                params['disLA'] = self.disLA.state_dict()
                params['disLB'] = self.disLB.state_dict()
                torch.save(params, os.path.join(self.result_dir, self.dataset, self.folder + '_params_latest.pt'))

    def save(self, dir, step):
        params = {}
        params['genA2B'] = self.genA2B.state_dict()
        params['genB2A'] = self.genB2A.state_dict()
        params['disGA'] = self.disGA.state_dict()
        params['disGB'] = self.disGB.state_dict()
        params['disLA'] = self.disLA.state_dict()
        params['disLB'] = self.disLB.state_dict()
        torch.save(params, os.path.join(dir, self.dataset.split('/')[0] + '_params_%07d.pt' % step))

    def load(self, dir, step):
        params = torch.load(os.path.join(dir, self.dataset.split('/')[0] + '_params_%07d.pt' % step))
        self.genA2B.load_state_dict(params['genA2B'])
        self.genB2A.load_state_dict(params['genB2A'])
        self.disGA.load_state_dict(params['disGA'])
        self.disGB.load_state_dict(params['disGB'])
        self.disLA.load_state_dict(params['disLA'])
        self.disLB.load_state_dict(params['disLB'])

    def val(self):
        list_model = sorted(
            glob(os.path.join(self.result_dir, self.dataset, self.folder, 'model' + str(self.stage), '*' + '.pt')))
        max_value = [0, 0, 0, 0, 0]
        for m in list_model:
            model_list = glob(
                os.path.join(self.result_dir, self.dataset, self.folder, 'model' + str(self.stage),
                             str(m.split('/')[-1].split('\'')[0])))
            model_name = str(m.split('/')[-1].split('\'')[0])
            if not len(model_list) == 0:
                model_list.sort()
                iter = int(model_list[-1].split('_')[-1].split('.')[0])
                self.load(os.path.join(self.result_dir, self.dataset, self.folder, 'model' + str(self.stage)), iter)
                print(" [*] Load SUCCESS")
            else:
                print(" [*] Load FAILURE")
                return

            self.genA2B.eval(), self.genB2A.eval()
            for n, (real_A, _, pathA) in enumerate(self.valA_loader):
                real_A = real_A.to(self.device)
                real_A2 = real_A

                fake_A2B, cam_logit, fake_A2B_heatmap, att_maskA2B, fake_A2B_2, real_A2_r = self.genA2B(real_A2)

                pathA = pathA[0].split('/')[-1].split('.')[0] + '.' + pathA[0].split('/')[-1].split('.')[1]


                path1 = os.path.join(self.result_dir, self.dataset, self.folder, 'val_folder',
                                     'results_mask_reverse' + str(self.stage), model_name)
                path2 = os.path.join(self.result_dir, self.dataset, self.folder, 'val_folder',
                                     'iccv_labels' + str(self.stage), model_name)
                path3 = os.path.join(self.result_dir, self.dataset, self.folder, 'val_folder',
                                     'results_mask' + str(self.stage), model_name)
                path4 = os.path.join(self.result_dir, self.dataset, self.folder, 'val_folder',
                                     'heatmap_A2B' + str(self.stage), model_name)
                path5 = os.path.join(self.result_dir, self.dataset, self.folder, 'val_folder',
                                     'fake_A2B' + str(self.stage), model_name)
                path6 = os.path.join(self.result_dir, self.dataset, self.folder, 'val_folder',
                                     'fake_A2B_2' + str(self.stage), model_name)

                if (os.path.exists(path1) != True):
                    os.makedirs(path1)
                if (os.path.exists(path2) != True):
                    os.makedirs(path2)
                if (os.path.exists(path3) != True):
                    os.makedirs(path3)
                if (os.path.exists(path4) != True):
                    os.makedirs(path4)
                if (os.path.exists(path5) != True):
                    os.makedirs(path5)
                if (os.path.exists(path6) != True):
                    os.makedirs(path6)

                background = torch.zeros_like(att_maskA2B).to(self.device)
                att_maskA2B = torch.where(real_A2 > -1, att_maskA2B, background)

                att_maskA2B_r = 1 - att_maskA2B
                MASKA2B_r = RGB2BGR(tensor2numpy(denorm(att_maskA2B_r[0])) * 2 - 1)
                if (self.dataset == "ISIC"):
                    pathA = pathA.split('.')[0]
                cv2.imwrite(os.path.join(path1, pathA + '_a2_b.png'), MASKA2B_r * 255.0)

                att_maskA2B = att_maskA2B
                MASKA2B = RGB2BGR(tensor2numpy(denorm(att_maskA2B[0])) * 2 - 1)
                cv2.imwrite(os.path.join(path3, pathA + '.png'), MASKA2B * 255.0)

                HEATMAP_A2B = cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size)
                cv2.imwrite(os.path.join(path4, pathA + '.png'), HEATMAP_A2B * 255.0)

                FAKEA2B = RGB2BGR(tensor2numpy(denorm(fake_A2B[0])))
                cv2.imwrite(os.path.join(path5, pathA + '.png'), FAKEA2B * 255.0)

                FAKEA2B_2 = RGB2BGR(tensor2numpy(denorm(fake_A2B_2[0])))
                cv2.imwrite(os.path.join(path6, pathA + '.png'), FAKEA2B_2 * 255.0)

            root_path = os.path.join(self.result_dir, self.dataset, self.folder)
            find_path = os.path.join(path1)
            find_output_path = os.path.join(path2)
            findCont(model_name, find_path, find_output_path)
            five_pre_path = os.path.join(path2)
            five_label_path = os.path.join('dataset', self.dataset, self.folder, 'val', 'labels')
            return_value = five_m(model_name, five_pre_path, five_label_path, self.dataset, self.folder, self.stage)
            if (return_value[0] > max_value[0]):
                max_value = return_value.copy()
                with open(root_path + "/val_max.txt", "w") as f:
                    f.write(self.dataset)
                    f.write('\n')
                    f.write(self.folder)
                    f.write('\n')
                    f.write(", ".join(str(i) for i in max_value))
                    f.write('\n')
                    f.write(model_name)
                    f.write('\n')
            print(" max_value: %s " % max_value)
            print(" %s Val finished!" % (model_name))

        print(" All Val finished!")
    def test(self):
        list_model = sorted(glob(os.path.join(self.result_dir, self.dataset, self.folder, 'model'+str(self.stage), '*' + '.pt')))
        max_value = [0, 0, 0, 0, 0]
        for m in list_model:
            model_list = glob(
                os.path.join(self.result_dir, self.dataset, self.folder, 'model'+str(self.stage), str(m.split('/')[-1].split('\'')[0])))
            model_name = str(m.split('/')[-1].split('\'')[0])
            if not len(model_list) == 0:
                model_list.sort()
                iter = int(model_list[-1].split('_')[-1].split('.')[0])
                self.load(os.path.join(self.result_dir, self.dataset, self.folder, 'model'+str(self.stage)), iter)
                print(" [*] Load SUCCESS")
            else:
                print(" [*] Load FAILURE")
                return

            self.genA2B.eval(), self.genB2A.eval()
            for n, (real_A, _, pathA) in enumerate(self.testA_loader):

                real_A = real_A.to(self.device)
                real_A2 = real_A

                fake_A2B, cam_logit, fake_A2B_heatmap, att_maskA2B, fake_A2B_2, real_A2_r = self.genA2B(real_A2)

                pathA = pathA[0].split('/')[-1].split('.')[0] + '.' +pathA[0].split('/')[-1].split('.')[1]

                path1 = os.path.join(self.result_dir, self.dataset, self.folder, 'results_mask_reverse'+str(self.stage), model_name)
                path2 = os.path.join(self.result_dir, self.dataset, self.folder, 'iccv_labels'+str(self.stage), model_name)
                path3 = os.path.join(self.result_dir, self.dataset, self.folder, 'results_mask' + str(self.stage), model_name)
                path4 = os.path.join(self.result_dir, self.dataset, self.folder, 'heatmap_A2B' + str(self.stage), model_name)
                path5 = os.path.join(self.result_dir, self.dataset, self.folder, 'fake_A2B' + str(self.stage), model_name)
                path6 = os.path.join(self.result_dir, self.dataset, self.folder, 'fake_A2B_2' + str(self.stage), model_name)

                if(os.path.exists(path1)!=True):
                    os.mkdir(path1)
                if(os.path.exists(path2)!=True):
                    os.mkdir(path2)
                if (os.path.exists(path3) != True):
                    os.mkdir(path3)
                if (os.path.exists(path4) != True):
                    os.mkdir(path4)
                if (os.path.exists(path5) != True):
                    os.mkdir(path5)
                if (os.path.exists(path6) != True):
                    os.mkdir(path6)

                background = torch.zeros_like(att_maskA2B).to(self.device)
                att_maskA2B = torch.where(real_A2 > -1, att_maskA2B, background)

                att_maskA2B_r = 1 - att_maskA2B
                MASKA2B_r = RGB2BGR(tensor2numpy(denorm(att_maskA2B_r[0]))*2-1)
                if(self.dataset=="ISIC"):
                    pathA = pathA.split('.')[0]
                cv2.imwrite(os.path.join(path1, pathA + '_a2_b.png'), MASKA2B_r * 255.0)

                att_maskA2B = att_maskA2B
                MASKA2B = RGB2BGR(tensor2numpy(denorm(att_maskA2B[0]))*2-1)
                cv2.imwrite(os.path.join(path3, pathA + '.png'), MASKA2B * 255.0)

                HEATMAP_A2B = cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size)
                cv2.imwrite(os.path.join(path4, pathA + '.png'), HEATMAP_A2B * 255.0)

                FAKEA2B = RGB2BGR(tensor2numpy(denorm(fake_A2B[0])))
                cv2.imwrite(os.path.join(path5, pathA + '.png'), FAKEA2B * 255.0)

                FAKEA2B_2 = RGB2BGR(tensor2numpy(denorm(fake_A2B_2[0])))
                cv2.imwrite(os.path.join(path6, pathA + '.png'), FAKEA2B_2 * 255.0)

            root_path = os.path.join(self.result_dir, self.dataset, self.folder)
            find_path = os.path.join(root_path, 'results_mask_reverse' + str(self.stage), model_name)
            find_output_path = os.path.join(root_path, 'iccv_labels' + str(self.stage), model_name)
            findCont(model_name, find_path, find_output_path)
            five_pre_path = os.path.join(root_path, 'iccv_labels' + str(self.stage), model_name)
            five_label_path = os.path.join('dataset', self.dataset, self.folder, 'test', 'labels')
            return_value = five_m(model_name, five_pre_path, five_label_path, self.dataset, self.folder, self.stage)
            if (return_value[0] > max_value[0]):
                max_value = return_value.copy()
                with open(root_path + "/max.txt", "w") as f:
                    f.write(self.dataset)
                    f.write('\n')
                    f.write(self.folder)
                    f.write('\n')
                    f.write(", ".join(str(i) for i in max_value))
                    f.write('\n')
                    f.write(model_name)
                    f.write('\n')
            print(" max_value: %s " % max_value)
            print(" %s Test finished!" % (model_name))

            for n, (real_B, _, pathB) in enumerate(self.testB_loader):
                real_B = real_B.to(self.device)

                fake_B2A, cam_logit, fake_B2A_heatmap, att_maskB2A, fake_B2A_2, real_B2_r = self.genA2B(real_B)

                pathB = pathB[0].split('/')[-1].split('.')[0] + '.' +pathB[0].split('/')[-1].split('.')[1]

                # path1 = "/home/dw/Disk_8T/SY/CBFNet-pytorch-master_pure/results/selfie2anime/results_mask_reverse" + m
                path1 = os.path.join(self.result_dir, self.dataset, self.folder, "folder_B2A", 'results_mask_reverse'+str(self.stage), model_name)
                # path2 = "/home/dw/Disk_8T/SY/CBFNet-pytorch-master_pure/iccv_labels" + '/' + 'iccv_labels' + m
                path2 = os.path.join(self.result_dir, self.dataset, self.folder,  "folder_B2A", 'iccv_labels'+str(self.stage), model_name)
                path3 = os.path.join(self.result_dir, self.dataset, self.folder,  "folder_B2A", 'results_mask' + str(self.stage), model_name)
                path4 = os.path.join(self.result_dir, self.dataset, self.folder,  "folder_B2A", 'heatmap_B2A' + str(self.stage), model_name)
                path5 = os.path.join(self.result_dir, self.dataset, self.folder,  "folder_B2A", 'fake_B2A' + str(self.stage), model_name)
                path6 = os.path.join(self.result_dir, self.dataset, self.folder,  "folder_B2A", 'fake_B2A_2' + str(self.stage), model_name)

                if(os.path.exists(path1)!=True):
                    os.makedirs(path1)
                if(os.path.exists(path2)!=True):
                    os.makedirs(path2)
                if (os.path.exists(path3) != True):
                    os.makedirs(path3)
                if (os.path.exists(path4) != True):
                    os.makedirs(path4)
                if (os.path.exists(path5) != True):
                    os.makedirs(path5)
                if (os.path.exists(path6) != True):
                    os.makedirs(path6)

                background = torch.zeros_like(att_maskB2A).to(self.device)
                att_maskB2A = torch.where(real_A > -1, att_maskB2A, background)

                att_maskB2A_r = 1 - att_maskB2A
                MASKB2A_r = RGB2BGR(tensor2numpy(denorm(att_maskB2A_r[0]))*2-1)
                if(self.dataset=="ISIC"):
                    pathB = pathB.split('.')[0]
                cv2.imwrite(os.path.join(path1, pathB + '_a2_b.png'), MASKB2A_r * 255.0)

                att_maskB2A = att_maskB2A
                MASKB2A = RGB2BGR(tensor2numpy(denorm(att_maskB2A[0]))*2-1)
                cv2.imwrite(os.path.join(path3, pathB + '.png'), MASKB2A * 255.0)

                HEATMAP_B2A = cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size)
                cv2.imwrite(os.path.join(path4, pathB + '.png'), HEATMAP_B2A * 255.0)

                FAKEB2A = RGB2BGR(tensor2numpy(denorm(fake_B2A[0])))
                cv2.imwrite(os.path.join(path5, pathB + '.png'), FAKEB2A * 255.0)

                FAKEB2A_2 = RGB2BGR(tensor2numpy(denorm(fake_B2A_2[0])))
                cv2.imwrite(os.path.join(path6, pathB + '.png'), FAKEB2A_2 * 255.0)
            print(" All B2A image produced!")



        print(" All Test finished!")

    def pseud_mask(self):
        list_model = sorted(glob(os.path.join(self.result_dir, self.dataset, self.folder, 'model'+str(self.stage), '*' + '.pt')))
        max_value = [0, 0, 0, 0, 0]
        for m in list_model:
            model_list = glob(
                os.path.join(self.result_dir, self.dataset, self.folder, 'model'+str(self.stage), str(m.split('/')[-1].split('\'')[0])))
            model_name = str(m.split('/')[-1].split('\'')[0])
            if not len(model_list) == 0:
                model_list.sort()
                iter = int(model_list[-1].split('_')[-1].split('.')[0])
                self.load(os.path.join(self.result_dir, self.dataset, self.folder, 'model'+str(self.stage)), iter)
                print(" [*] Load SUCCESS")
            else:
                print(" [*] Load FAILURE")
                return

            self.genA2B.eval(), self.genB2A.eval()
            for n, (real_A, _, pathA) in enumerate(self.pseud_A_loader):
                real_A = real_A.to(self.device)
                real_A2 = real_A

                fake_A2B, _, fake_A2B_heatmap, att_maskA2B, _, _ = self.genA2B(real_A2)

                pathA = pathA[0].split('/')[-1].split('.')[0] + '.' +pathA[0].split('/')[-1].split('.')[1]

                path1 = os.path.join(self.result_dir, self.dataset, self.folder, 'pseud_results_mask_reverse'+str(self.stage), model_name)
                path2 = os.path.join(self.result_dir, self.dataset, self.folder, 'pseud_iccv_labels'+str(self.stage), model_name)
                path3 = os.path.join(self.result_dir, self.dataset, self.folder, 'pseud_results_mask' + str(self.stage), model_name)

                if(os.path.exists(path1)!=True):
                    os.mkdir(path1)
                if(os.path.exists(path2)!=True):
                    os.mkdir(path2)
                if (os.path.exists(path3) != True):
                    os.mkdir(path3)

                background = torch.zeros_like(att_maskA2B).to(self.device)
                att_maskA2B = torch.where(real_A2 > -1, att_maskA2B, background)

                att_maskA2B_r = 1 - att_maskA2B
                MASKA2B_r = RGB2BGR(tensor2numpy(denorm(att_maskA2B_r[0]))*2-1)
                if(self.dataset=="ISIC"):
                    pathA = pathA.split('.')[0]
                cv2.imwrite(os.path.join(path1, pathA + '_a2_b.png'), MASKA2B_r * 255.0)

                att_maskA2B = att_maskA2B
                MASKA2B = RGB2BGR(tensor2numpy(denorm(att_maskA2B[0]))*2-1)
                cv2.imwrite(os.path.join(path3, pathA + '.png'), MASKA2B * 255.0)

        print(" All pseud mask produced!")

