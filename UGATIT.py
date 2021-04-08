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
from five_metrics import five_m_val

class UGATIT(object) :
    def __init__(self, args):
        self.light = args.light

        if self.light :
            self.model_name = 'UGATIT_light'
        else :
            self.model_name = 'UGATIT'

        self.result_dir = args.result_dir
        self.dataset = args.dataset

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

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.device = args.device
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume
        self.imglist_testA = os.listdir(os.path.join('dataset', self.dataset, 'testA'))
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
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(p=0.5),
            # transforms.ColorJitter( contrast=(0, 10)),
            # transforms.Resize((self.img_size + 30, self.img_size+30)),
            # transforms.RandomCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        test_transform = transforms.Compose([
            # transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.trainA = ImageFolder(os.path.join('dataset', self.dataset, 'trainA'), train_transform)
        self.trainB = ImageFolder(os.path.join('dataset', self.dataset, 'trainB'), train_transform)
        self.testA = ImageFolder(os.path.join('dataset', self.dataset, 'testA'), test_transform)
        self.testB = ImageFolder(os.path.join('dataset', self.dataset, 'testB'), test_transform)
        self.trainA_iccv = ImageFolder(os.path.join('dataset', self.dataset, 'trainA'), train_transform)
        self.valA = ImageFolder(os.path.join('dataset', self.dataset, 'val', 'images'), test_transform)

        self.trainA_loader = DataLoader(self.trainA, batch_size=self.batch_size, shuffle=True)
        self.trainB_loader = DataLoader(self.trainB, batch_size=self.batch_size, shuffle=True)
        self.testA_loader = DataLoader(self.testA, batch_size=1, shuffle=False)
        self.testB_loader = DataLoader(self.testB, batch_size=1, shuffle=False)
        self.trainA_iccv_loader = DataLoader(self.trainA_iccv, batch_size=1, shuffle=True)
        self.valA_loader = DataLoader(self.valA, batch_size=1, shuffle=False)

        """ Define Generator, Discriminator """
        self.genA2B = ResnetGenerator(input_nc=6, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
        self.genB2A = ResnetGenerator(input_nc=6, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
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
            model_list = glob(os.path.join(self.result_dir, self.dataset, 'model'+str(self.stage), '*.pt'))
            if not len(model_list) == 0:
                model_list.sort()
                start_iter = int(model_list[-1].split('_')[-1].split('.')[0])
                self.load(os.path.join(self.result_dir, self.dataset, 'model'+str(self.stage)), start_iter)
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


            # if (randint(0, 10) % 2 == 0):
            real_A2 = torch.cat([real_A, realA_out], 1)

            # imglist_trainB = os.listdir(os.path.join('dataset', self.dataset, 'syn_trainB'))
            # shuffle(imglist_trainB)
            # for t in range(self.batch_size):
            #     # for img in imglist_trainB:
            #     img = str(pathB).split('/')[-1].split('\'')[0]
            #     tpath = os.path.join('dataset', self.dataset, 'syn_trainB', img)
            #     fopen = Image.open(tpath)
            #     transform = transforms.Compose(
            #         [transforms.ToTensor(),
            #          transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
            #     real_B1 = transform(fopen)  # data就是预处理后，可以送入模型进行训练的数据了
            #     real_B1 = real_B1.unsqueeze(0)
            #     if (t == 0):
            #         realB_out = real_B1
            #     else:
            #         realB_out = torch.cat([realB_out, real_B1], 0)
            #     break

            # if (randint(0, 10) % 2 == 0):
            realB_out = real_B
            real_B2 = torch.cat([real_B, realB_out], 1)

            real_A, real_B = real_A.to(self.device), real_B.to(self.device)
            real_A2, real_B2 = real_A2.to(self.device), real_B2.to(self.device)

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

            Discriminator_loss = 15 * (D_loss_A + D_loss_B)
            Discriminator_loss.backward()
            self.D_optim.step()

            # Update G
            self.G_optim.zero_grad()

            fake_A2B, fake_A2B_cam_logit, fake_A2B_heatmap, att_maskA2B, contentA2B, realA_r = self.genA2B(real_A2)
            fake_B2A, fake_B2A_cam_logit, fake_B2A_heatmap, att_maskB2A, contentB2A, realB_r = self.genB2A(real_B2)

            # fake_input_A = real_A*att_maskA2B + contentA2B*(1-att_maskA2B)
            # fake_input_B = real_B*att_maskB2A + contentB2A*(1-att_maskB2A)

            fake_B2A1 = fake_B2A.clone().detach()
            fake_B2A2 = torch.cat([fake_B2A, fake_B2A1], 1)
            fake_B2A2B, _, _, _, _, _ = self.genA2B(fake_B2A2)
            fake_A2B1 = fake_A2B.clone().detach()
            fake_A2B2 = torch.cat([fake_A2B, fake_A2B1], 1)
            fake_A2B2A, _, _, _, _, _ = self.genB2A(fake_A2B2)

            fake_B2B, fake_B2B_cam_logit, _, _, content_B2B, _ = self.genA2B(real_B2)
            fake_A2A, fake_A2A_cam_logit, _, _, _, _ = self.genB2A(real_A2)

            fake_GA_logit_r, fake_GA_cam_logit_r, _ = self.disGA(realA_r)
            fake_LA_logit_r, fake_LA_cam_logit_r, _ = self.disLA(realA_r)
            fake_GB_logit_r, fake_GB_cam_logit_r, _ = self.disGB(realB_r)
            fake_LB_logit_r, fake_LB_cam_logit_r, _ = self.disLB(realB_r)

            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
            fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
            fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

            # fake_GA_logit_f, fake_GA_cam_logit_f, _ = self.disGA(fake_input_A)
            # fake_LA_logit_f, fake_LA_cam_logit_f, _ = self.disLA(fake_input_A)
            # fake_GB_logit_f, fake_GB_cam_logit_f, _ = self.disGB(fake_input_B)
            # fake_LB_logit_f, fake_LB_cam_logit_f, _ = self.disLB(fake_input_B)



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



            # G_ad_loss_GA_f = self.MSE_loss(fake_GA_logit_f, torch.ones_like(fake_GA_logit_f).to(self.device))
            # G_ad_cam_loss_GA_f = self.MSE_loss(fake_GA_cam_logit_f, torch.ones_like(fake_GA_cam_logit_f).to(self.device))
            # G_ad_loss_LA_f = self.MSE_loss(fake_LA_logit_f, torch.ones_like(fake_LA_logit_f).to(self.device))
            # G_ad_cam_loss_LA_f = self.MSE_loss(fake_LA_cam_logit_f, torch.ones_like(fake_LA_cam_logit_f).to(self.device))
            # G_ad_loss_GB_f = self.MSE_loss(fake_GB_logit_f, torch.ones_like(fake_GB_logit_f).to(self.device))
            # G_ad_cam_loss_GB_f = self.MSE_loss(fake_GB_cam_logit_f, torch.ones_like(fake_GB_cam_logit_f).to(self.device))
            # G_ad_loss_LB_f = self.MSE_loss(fake_LB_logit_f, torch.ones_like(fake_LB_logit_f).to(self.device))
            # G_ad_cam_loss_LB_f = self.MSE_loss(fake_LB_cam_logit_f, torch.ones_like(fake_LB_cam_logit_f).to(self.device))
            # adv_loss_A_f = G_ad_loss_GA_f + G_ad_cam_loss_GA_f + G_ad_loss_LA_f + G_ad_cam_loss_LA_f
            # adv_loss_B_f = G_ad_loss_GB_f + G_ad_cam_loss_GB_f + G_ad_loss_LB_f + G_ad_cam_loss_LB_f

            G_r_loss_A = self.L1_loss(realA_r, real_A)
            G_r_loss_B = self.L1_loss(realB_r, real_B)

            # G_mask_loss_A = self.L1_loss(att_maskA2B, real_A2[:, 3:6, :, :])

            G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
            G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)

            G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
            G_identity_loss_B = self.L1_loss(fake_B2B, real_B)

            G_cam_loss_A = self.BCE_loss(fake_B2A_cam_logit, torch.ones_like(fake_B2A_cam_logit).to(self.device)) + self.BCE_loss(fake_A2A_cam_logit, torch.zeros_like(fake_A2A_cam_logit).to(self.device))
            G_cam_loss_B = self.BCE_loss(fake_A2B_cam_logit, torch.ones_like(fake_A2B_cam_logit).to(self.device)) + self.BCE_loss(fake_B2B_cam_logit, torch.zeros_like(fake_B2B_cam_logit).to(self.device))

            G_loss_A = self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + self.cycle_weight * G_recon_loss_A + self.identity_weight * G_identity_loss_A + self.cam_weight * G_cam_loss_A# + 0.7*G_r_loss_A #+ adv_loss_A_f# + G_mask_loss_A
            # G_loss_A = self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + self.cycle_weight * G_recon_loss_A + self.cam_weight * G_cam_loss_A
            G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB ) + self.cycle_weight * G_recon_loss_B + self.identity_weight * G_identity_loss_B + self.cam_weight * G_cam_loss_B# + 0.7*G_r_loss_B #+ adv_loss_B_f
            # G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + self.cycle_weight * G_recon_loss_B + self.cam_weight * G_cam_loss_B

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

            # for t in range(self.batch_size):
            #     MASKA2B = RGB2BGR(tensor2numpy(denorm(att_maskA2B[t])))
            #     MASKB2A = RGB2BGR(tensor2numpy(denorm(att_maskB2A[t])))
            #     MASKA2B_name = str(pathA[t]).split('/')[-1]
            #     MASKB2A_name = str(pathB[t]).split('/')[-1]
            #     cv2.imwrite(os.path.join('dataset', self.dataset, 'syn_trainA', MASKA2B_name), MASKA2B * 255.0)
            #     cv2.imwrite(os.path.join('dataset', self.dataset, 'syn_trainB', MASKB2A_name), MASKB2A * 255.0)

            if step % self.print_freq == 0:
                train_sample_num = 18
                # test_sample_num = 5
                A2B = np.zeros((self.img_size * 6, 0, 3))
                # B2A = np.zeros((self.img_size * 7, 0, 3))

                self.genA2B.eval(), self.genB2A.eval(), self.disGA.eval(), self.disGB.eval(), self.disLA.eval(), self.disLB.eval()
                for _ in range(train_sample_num):
                    # try:
                    #     real_A, _ = trainA_iter.next()
                    # except:
                    #     trainA_iter = iter(self.trainA_loader)
                    #     real_A, _ = trainA_iter.next()
                    #
                    # try:
                    #     real_B, _ = trainB_iter.next()
                    # except:
                    #     trainB_iter = iter(self.trainB_loader)
                    #     real_B, _ = trainB_iter.next()

                    imgA = imglist_testA[test_start % len(imglist_testA)]
                    tpath = os.path.join('dataset', self.dataset, 'testA', imgA)
                    fopen = Image.open(tpath)
                    transform = transforms.Compose(
                        [transforms.ToTensor(),
                         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
                    real_A = transform(fopen)  # data就是预处理后，可以送入模型进行训练的数据了
                    real_A = real_A.unsqueeze(0)

                        # for img in imglist_trainA:
                    if(self.stage==2):
                        tpath = os.path.join('dataset', self.dataset, 'syn_testA', imgA.split('.png')[
                            0] + '.png')  # 路径(/home/ouc/river/test)+图片名（img_m）
                        fopen = Image.open(tpath)
                        transform = transforms.Compose(
                            [transforms.ToTensor(),
                             transforms.Normalize(mean=(0.5), std=(0.5))])
                        real_A1 = transform(fopen)  # data就是预处理后，可以送入模型进行训练的数据
                        # real_A1 = torch.cat([real_A1,real_A1,real_A1],0)
                        real_A1 = real_A1.unsqueeze(0)
                        realA_out = real_A1
                    else:
                        realA_out = real_A

                    # realA_out = realA_out.to(self.device)
                    real_A2 = torch.cat([real_A, realA_out], 1)

                    imgB = imglist_testB[test_start % len(imglist_testB)]
                    tpath = os.path.join('dataset', self.dataset, 'testB', imgB)
                    fopen = Image.open(tpath)
                    transform = transforms.Compose(
                        [transforms.ToTensor(),
                         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
                    real_B = transform(fopen)  # data就是预处理后，可以送入模型进行训练的数据了
                    real_B = real_B.unsqueeze(0)
                    real_B1 = real_B.clone()
                    real_B2 = torch.cat([real_B, real_B1], 1)

                    real_A, real_B = real_A.to(self.device), real_B.to(self.device)
                    real_A2, real_B2 = real_A2.to(self.device), real_B2.to(self.device)
                    test_start += 1

                    fake_A2B, _, fake_A2B_heatmap, att_maskA2B, contentA2B, real_A2_r = self.genA2B(real_A2)
                    fake_B2A, _, fake_B2A_heatmap, att_maskB2A, contentB2A, real_B2_r = self.genB2A(real_B2)

                    A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                               cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(att_maskA2B[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(contentA2B[0]))),
                                                               RGB2BGR(tensor2numpy(denorm(real_A2_r[0])))), 0)), 1)

                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img'+str(self.stage), 'A2B_%07d.png' % step), A2B * 255.0)
                # cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'B2A_%07d.png' % step), B2A * 255.0)
                self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

            if step % self.save_freq == 0:
                self.save(os.path.join(self.result_dir, self.dataset, 'model'+str(self.stage)), step)

            if step % 1000 == 0:
                params = {}
                params['genA2B'] = self.genA2B.state_dict()
                params['genB2A'] = self.genB2A.state_dict()
                params['disGA'] = self.disGA.state_dict()
                params['disGB'] = self.disGB.state_dict()
                params['disLA'] = self.disLA.state_dict()
                params['disLB'] = self.disLB.state_dict()
                torch.save(params, os.path.join(self.result_dir, self.dataset + '_params_latest.pt'))

    def save(self, dir, step):
        params = {}
        params['genA2B'] = self.genA2B.state_dict()
        params['genB2A'] = self.genB2A.state_dict()
        params['disGA'] = self.disGA.state_dict()
        params['disGB'] = self.disGB.state_dict()
        params['disLA'] = self.disLA.state_dict()
        params['disLB'] = self.disLB.state_dict()
        # torch.save(params, os.path.join(dir, self.dataset + '_params_%07d.pt' % step))
        torch.save(params, os.path.join(dir, self.dataset.split('/')[0] + '_params_%07d.pt' % step))

    def load(self, dir, step):
        params = torch.load(os.path.join(dir, self.dataset.split('/')[0] + '_params_%07d.pt' % step))
        self.genA2B.load_state_dict(params['genA2B'])
        self.genB2A.load_state_dict(params['genB2A'])
        self.disGA.load_state_dict(params['disGA'])
        self.disGB.load_state_dict(params['disGB'])
        self.disLA.load_state_dict(params['disLA'])
        self.disLB.load_state_dict(params['disLB'])

    def test(self):
        # list_model = ['7920', '8010','8040','8070','8130','8160','8190','8220','8250','8430','8700',
        #               '8730','8920','9480','9600','9720','9750','9780','9840','9870','9900',
        #               '9930','9960','9990','9930','10020','10050','10080','10200','10230']
        # list_model = ['2140','2180','5080','5100','5120','5950','6000','6050','6150','6200','6900','7700',
        #               '7850', '7900','7950']
        list_model = sorted(glob(os.path.join(self.result_dir, self.dataset, 'model'+str(self.stage), '*' + '.pt')))
        for m in list_model:
            model_list = glob(
                os.path.join(self.result_dir, self.dataset, 'model'+str(self.stage), str(m.split('/')[-1].split('\'')[0])))
            k = str(m.split('/')[-1].split('\'')[0])
            if not len(model_list) == 0:
                model_list.sort()
                iter = int(model_list[-1].split('_')[-1].split('.')[0])
                self.load(os.path.join(self.result_dir, self.dataset, 'model'+str(self.stage)), iter)
                print(" [*] Load SUCCESS")
            else:
                print(" [*] Load FAILURE")
                return

            self.genA2B.eval(), self.genB2A.eval()
            for n, (real_A, _, pathA) in enumerate(self.testA_loader):
                real_A = real_A.to(self.device)

                if(self.stage==2):
                    # for img in imglist_trainA:
                    img = str(pathA[0]).split('/')[-1]
                    tpath = os.path.join('dataset', self.dataset, 'syn_testA', img.split('.png')[
                        0] + '.png')  # 路径(/home/ouc/river/test)+图片名（img_m）
                    fopen = Image.open(tpath)
                    transform = transforms.Compose(
                        [transforms.ToTensor(),
                         transforms.Normalize(mean=(0.5), std=(0.5))])
                    real_A1 = transform(fopen)  # data就是预处理后，可以送入模型进行训练的数据
                    # real_A1 = torch.cat([real_A1,real_A1,real_A1],0)
                    real_A1 = real_A1.unsqueeze(0)
                    realA_out = real_A1

                else:
                    realA_out = real_A

                realA_out = realA_out.to(self.device)

                real_A2 = torch.cat([real_A, realA_out], 1)

                fake_A2B, _, fake_A2B_heatmap, att_maskA2B, _, _ = self.genA2B(real_A2)

                pathA = pathA[0].split('/')[-1].split('.')[0] + '.' +pathA[0].split('/')[-1].split('.')[1]

                # path1 = "/home/dw/Disk_8T/SY/UGATIT-pytorch-master_pure/results/selfie2anime/results_mask_reverse" + m
                path1 = os.path.join(self.result_dir, self.dataset, 'results_mask_reverse'+str(self.stage), k)
                # path2 = "/home/dw/Disk_8T/SY/UGATIT-pytorch-master_pure/iccv_labels" + '/' + 'iccv_labels' + m
                path2 = os.path.join(self.result_dir, self.dataset, 'iccv_labels'+str(self.stage), k)
                path3 = os.path.join(self.result_dir, self.dataset, 'results_mask' + str(self.stage), k)

                if(os.path.exists(path1)!=True):
                    os.mkdir(path1)
                if(os.path.exists(path2)!=True):
                    os.mkdir(path2)
                if (os.path.exists(path3) != True):
                    os.mkdir(path3)

                # heatmap_fake_A2B = cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size)
                # cv2.imwrite(
                #     os.path.join(self.result_dir, self.dataset, '555_heatmap', pathA.split('.png')[0] + '_a2_b.png'),
                #     heatmap_fake_A2B * 255.0)
                att_maskA2B_r = 1 - att_maskA2B
                MASKA2B_r = RGB2BGR(tensor2numpy(denorm(att_maskA2B_r[0])))
                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'results_mask_reverse'+str(self.stage), k, pathA + '_a2_b.png'), MASKA2B_r * 255.0)

                att_maskA2B = att_maskA2B
                MASKA2B = RGB2BGR(tensor2numpy(denorm(att_maskA2B[0])))
                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'results_mask' + str(self.stage), k, pathA + '.png'),
                            MASKA2B * 255.0)

            if(self.stage==1):
                for n, (real_A, _, pathA) in enumerate(self.trainA_iccv_loader):
                    real_A = real_A.to(self.device)
                    print(n)
                    if (self.stage == 2):
                        # for img in imglist_trainA:
                        img = str(pathA[0]).split('/')[-1]
                        tpath = os.path.join('dataset', self.dataset, 'syn_trainA', img.split('.png')[
                            0] + '.png')  # 路径(/home/ouc/river/test)+图片名（img_m）
                        fopen = Image.open(tpath)
                        transform = transforms.Compose(
                            [transforms.ToTensor(),
                             transforms.Normalize(mean=(0.5), std=(0.5))])
                        real_A1 = transform(fopen)  # data就是预处理后，可以送入模型进行训练的数据
                        # real_A1 = torch.cat([real_A1,real_A1,real_A1],0)
                        real_A1 = real_A1.unsqueeze(0)
                        realA_out = real_A1

                    else:
                        realA_out = real_A

                    realA_out = realA_out.to(self.device)

                    real_A2 = torch.cat([real_A, realA_out], 1)

                    fake_A2B, _, fake_A2B_heatmap, att_maskA2B, _, _ = self.genA2B(real_A2)

                    pathA = pathA[0].split('/')[-1].split('.')[0] + '.' + pathA[0].split('/')[-1].split('.')[1]

                    # path1 = "/home/dw/Disk_8T/SY/UGATIT-pytorch-master_pure/results/selfie2anime/results_mask_reverse" + m
                    path1 = os.path.join(self.result_dir, self.dataset, 'trainA_labels' + str(self.stage), k)
                    # path2 = "/home/dw/Disk_8T/SY/UGATIT-pytorch-master_pure/iccv_labels" + '/' + 'iccv_labels' + m
                    # path2 = os.path.join(self.result_dir, self.dataset, 'iccv_labels' + str(self.stage), k)
                    # path3 = os.path.join(self.result_dir, self.dataset, 'results_mask' + str(self.stage), k)

                    if (os.path.exists(path1) != True):
                        os.mkdir(path1)
                    # if (os.path.exists(path2) != True):
                    #     os.mkdir(path2)
                    # if (os.path.exists(path3) != True):
                    #     os.mkdir(path3)

                    # heatmap_fake_A2B = cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size)
                    # cv2.imwrite(
                    #     os.path.join(self.result_dir, self.dataset, '555_heatmap', pathA.split('.png')[0] + '_a2_b.png'),
                    #     heatmap_fake_A2B * 255.0)
                    # att_maskA2B_r = 1 - att_maskA2B
                    # MASKA2B_r = RGB2BGR(tensor2numpy(denorm(att_maskA2B_r[0])))
                    # cv2.imwrite(os.path.join(path1, pathA + '.png'), MASKA2B_r * 255.0)

                    att_maskA2B = att_maskA2B
                    MASKA2B = RGB2BGR(tensor2numpy(denorm(att_maskA2B[0])))
                    cv2.imwrite(os.path.join(path1,pathA + '.png'), MASKA2B * 255.0)



            find_path = os.path.join(self.result_dir, self.dataset, 'results_mask_reverse'+str(self.stage), k)
            find_output_path = os.path.join(self.result_dir, self.dataset, 'iccv_labels'+str(self.stage), k)
            findCont(k, find_path, find_output_path)
            five_pre_path = os.path.join(self.result_dir, self.dataset, 'iccv_labels'+str(self.stage), k)
            five_label_path = os.path.join('dataset', self.dataset, 'test', 'labels')
            five_m(k, five_pre_path, five_label_path, self.dataset, self.stage)
            print(" [*] Test finished!")

    def val(self):
        # list_model = ['7920', '8010','8040','8070','8130','8160','8190','8220','8250','8430','8700',
        #               '8730','8920','9480','9600','9720','9750','9780','9840','9870','9900',
        #               '9930','9960','9990','9930','10020','10050','10080','10200','10230']
        # list_model = ['2140','2180','5080','5100','5120','5950','6000','6050','6150','6200','6900','7700',
        #               '7850', '7900','7950']
        list_model = sorted(
            glob(os.path.join(self.result_dir, self.dataset, 'model' + str(self.stage), '*' + '.pt')))
        for m in list_model:
            model_list = glob(
                os.path.join(self.result_dir, self.dataset, 'model' + str(self.stage),
                             str(m.split('/')[-1].split('\'')[0])))
            k = str(m.split('/')[-1].split('\'')[0])
            if not len(model_list) == 0:
                model_list.sort()
                iter = int(model_list[-1].split('_')[-1].split('.')[0])
                self.load(os.path.join(self.result_dir, self.dataset, 'model' + str(self.stage)), iter)
                print(" [*] Load SUCCESS")
            else:
                print(" [*] Load FAILURE")
                return

            self.genA2B.eval(), self.genB2A.eval()
            for n, (real_A, _, pathA) in enumerate(self.valA_loader):
                real_A = real_A.to(self.device)

                if (self.stage == 2):
                    # for img in imglist_trainA:
                    img = str(pathA[0]).split('/')[-1]
                    tpath = os.path.join('dataset', self.dataset, 'syn_valA', img.split('.png')[
                        0] + '.png')  # 路径(/home/ouc/river/test)+图片名（img_m）
                    fopen = Image.open(tpath)
                    transform = transforms.Compose(
                        [transforms.ToTensor(),
                         transforms.Normalize(mean=(0.5), std=(0.5))])
                    real_A1 = transform(fopen)  # data就是预处理后，可以送入模型进行训练的数据
                    # real_A1 = torch.cat([real_A1,real_A1,real_A1],0)
                    real_A1 = real_A1.unsqueeze(0)
                    realA_out = real_A1

                else:
                    realA_out = real_A

                realA_out = realA_out.to(self.device)

                real_A2 = torch.cat([real_A, realA_out], 1)

                fake_A2B, _, fake_A2B_heatmap, att_maskA2B, _, _ = self.genA2B(real_A2)

                pathA = pathA[0].split('/')[-1].split('.')[0] + '.' + pathA[0].split('/')[-1].split('.')[1]

                # path1 = "/home/dw/Disk_8T/SY/UGATIT-pytorch-master_pure/results/selfie2anime/results_mask_reverse" + m
                path1 = os.path.join(self.result_dir, self.dataset, 'results_mask_reverse' + str(self.stage) + 'val', k)
                # path2 = "/home/dw/Disk_8T/SY/UGATIT-pytorch-master_pure/iccv_labels" + '/' + 'iccv_labels' + m
                path2 = os.path.join(self.result_dir, self.dataset, 'iccv_labels' + str(self.stage) + 'val', k)
                path3 = os.path.join(self.result_dir, self.dataset, 'results_mask' + str(self.stage) + 'val', k)

                if (os.path.exists(path1) != True):
                    os.mkdir(path1)
                if (os.path.exists(path2) != True):
                    os.mkdir(path2)
                if (os.path.exists(path3) != True):
                    os.mkdir(path3)

                # heatmap_fake_A2B = cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size)
                # cv2.imwrite(
                #     os.path.join(self.result_dir, self.dataset, '555_heatmap', pathA.split('.png')[0] + '_a2_b.png'),
                #     heatmap_fake_A2B * 255.0)
                att_maskA2B_r = 1 - att_maskA2B
                MASKA2B_r = RGB2BGR(tensor2numpy(denorm(att_maskA2B_r[0])))
                cv2.imwrite(
                    os.path.join(self.result_dir, self.dataset, 'results_mask_reverse' + str(self.stage) + 'val', k,
                                 pathA + '_a2_b.png'), MASKA2B_r * 255.0)

                att_maskA2B = att_maskA2B
                MASKA2B = RGB2BGR(tensor2numpy(denorm(att_maskA2B[0])))
                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'results_mask' + str(self.stage) + 'val', k,
                                         pathA + '.png'),
                            MASKA2B * 255.0)

            if (self.stage == 1):
                for n, (real_A, _, pathA) in enumerate(self.trainA_iccv_loader):
                    real_A = real_A.to(self.device)
                    if (self.stage == 2):
                        # for img in imglist_trainA:
                        img = str(pathA[0]).split('/')[-1]
                        tpath = os.path.join('dataset', self.dataset, 'syn_valA', img.split('.png')[
                            0] + '.png')  # 路径(/home/ouc/river/test)+图片名（img_m）
                        fopen = Image.open(tpath)
                        transform = transforms.Compose(
                            [transforms.ToTensor(),
                             transforms.Normalize(mean=(0.5), std=(0.5))])
                        real_A1 = transform(fopen)  # data就是预处理后，可以送入模型进行训练的数据
                        # real_A1 = torch.cat([real_A1,real_A1,real_A1],0)
                        real_A1 = real_A1.unsqueeze(0)
                        realA_out = real_A1

                    else:
                        realA_out = real_A

                    realA_out = realA_out.to(self.device)

                    real_A2 = torch.cat([real_A, realA_out], 1)

                    fake_A2B, _, fake_A2B_heatmap, att_maskA2B, _, _ = self.genA2B(real_A2)

                    pathA = pathA[0].split('/')[-1].split('.')[0] + '.' + pathA[0].split('/')[-1].split('.')[1]

                    # path1 = "/home/dw/Disk_8T/SY/UGATIT-pytorch-master_pure/results/selfie2anime/results_mask_reverse" + m
                    path1 = os.path.join(self.result_dir, self.dataset, 'trainA_labels' + str(self.stage) + 'val', k)
                    # path2 = "/home/dw/Disk_8T/SY/UGATIT-pytorch-master_pure/iccv_labels" + '/' + 'iccv_labels' + m
                    # path2 = os.path.join(self.result_dir, self.dataset, 'iccv_labels' + str(self.stage), k)
                    # path3 = os.path.join(self.result_dir, self.dataset, 'results_mask' + str(self.stage), k)

                    if (os.path.exists(path1) != True):
                        os.mkdir(path1)
                    # if (os.path.exists(path2) != True):
                    #     os.mkdir(path2)
                    # if (os.path.exists(path3) != True):
                    #     os.mkdir(path3)

                    # heatmap_fake_A2B = cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size)
                    # cv2.imwrite(
                    #     os.path.join(self.result_dir, self.dataset, '555_heatmap', pathA.split('.png')[0] + '_a2_b.png'),
                    #     heatmap_fake_A2B * 255.0)
                    # att_maskA2B_r = 1 - att_maskA2B
                    # MASKA2B_r = RGB2BGR(tensor2numpy(denorm(att_maskA2B_r[0])))
                    # cv2.imwrite(os.path.join(path1, pathA + '.png'), MASKA2B_r * 255.0)

                    att_maskA2B = att_maskA2B
                    MASKA2B = RGB2BGR(tensor2numpy(denorm(att_maskA2B[0])))
                    cv2.imwrite(os.path.join(path1, pathA + '.png'), MASKA2B * 255.0)

            find_path = os.path.join(self.result_dir, self.dataset, 'results_mask_reverse' + str(self.stage) + 'val', k)
            find_output_path = os.path.join(self.result_dir, self.dataset, 'iccv_labels' + str(self.stage) + 'val', k)
            findCont(k, find_path, find_output_path)
            five_pre_path = os.path.join(self.result_dir, self.dataset, 'iccv_labels' + str(self.stage) + 'val', k)
            five_label_path = os.path.join('dataset', self.dataset, 'val', 'labels')
            five_m_val(k, five_pre_path, five_label_path, self.dataset, self.stage)
            print(" [*] Test finished!")