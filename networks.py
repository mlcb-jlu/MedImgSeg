import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

        DownBlock = []
        DownBlock += [nn.ReflectionPad2d(3),
                      nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=False),
                      nn.InstanceNorm2d(ngf),
                      nn.ReLU(True)]

        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            DownBlock += [nn.ReflectionPad2d(1),
                          nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0, bias=False),
                          nn.InstanceNorm2d(ngf * mult * 2),
                          nn.ReLU(True)]

        # Down-Sampling Bottleneck
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        # Class Activation Map
        self.gap_fc = nn.Linear(ngf * mult, 1, bias=False)
        # self.gap_fc.requires_grad_(False)
        self.gmp_fc = nn.Linear(ngf * mult, 1, bias=False)
        # self.gmp_fc.requires_grad_(False)
        self.conv1x1 = nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=1, stride=1, bias=True)
        # self.conv2x1 = nn.Conv2d(ngf * mult, 1, kernel_size=1, stride=1, bias=True)
        conv2x1 = []
        conv2x1 += [  # ResnetBlock(ngf * mult, False),
            nn.ReflectionPad2d(1),
            # nn.Conv2d(ngf * mult, int(ngf * mult), kernel_size=3, stride=1, padding=0, bias=False),
            nn.Conv2d(int(ngf * mult), 1, kernel_size=3, stride=1, padding=0, bias=False),
        ]
        # self.conv1x1.requires_grad_(False)
        self.relu = nn.ReLU(True)

        # Gamma, Beta block
        if self.light:
            FC = [nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True),
                  nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True)]
        else:
            FC = [nn.Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True),
                  nn.Linear(ngf * mult, ngf * mult, bias=False),
                  nn.ReLU(True)]
        self.gamma1 = nn.Linear(ngf * mult, ngf * mult, bias=False)
        self.beta1 = nn.Linear(ngf * mult, ngf * mult, bias=False)
        self.gamma2 = nn.Linear(ngf * mult, ngf * mult, bias=False)
        self.beta2 = nn.Linear(ngf * mult, ngf * mult, bias=False)

        # Up-Sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i + 1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # for i in range(n_blocks):
        #     setattr(self, 'UpBlock2_' + str(i+1), ResnetAdaILNBlock(ngf * mult, use_bias=False))
        #
        for i in range(n_blocks):
            setattr(self, 'Mask_Layer_' + str(i + 1), Mask_Layer(ngf * mult * 2, use_bias=False))

        for i in range(n_blocks):
            setattr(self, 'recon_middle_' + str(i + 1), Recon_Middle(ngf * mult, use_bias=False))

        for i in range(n_blocks):
            setattr(self, 'Mask_Layer_UP_' + str(i + 1), Mask_Layer(ngf * mult * 2, use_bias=False))
        for i in range(n_blocks):
            setattr(self, 'Mask_Layer_DOWN_' + str(i + 1), Mask_Layer(ngf * mult * 2, use_bias=False))

        # Up-Sampling
        Decode_image = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            Decode_image += [nn.Upsample(scale_factor=2, mode='nearest'),
                             nn.ReflectionPad2d(1),
                             nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=False),
                             ILN(int(ngf * mult / 2)),
                             nn.ReLU(True)]

        Decode_image += [nn.ReflectionPad2d(1),
                         nn.Conv2d(ngf, output_nc, kernel_size=3, stride=1, padding=0, bias=False),
                         nn.Tanh()]

        Decode_recon = []
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            Decode_recon += [nn.Upsample(scale_factor=2, mode='nearest'),
                             nn.ReflectionPad2d(1),
                             nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=False),
                             ILN(int(ngf * mult / 2)),
                             nn.ReLU(True)]

        Decode_recon += [nn.ReflectionPad2d(3),
                         nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
                         nn.Tanh()]

        Split_layer = []
        mult = 4
        Split_layer += [nn.Conv2d(ngf * mult, ngf * mult, kernel_size=1, stride=1, padding=0, bias=False),
                        ILN(int(ngf * mult)),
                        nn.ReLU(True)]
        Split_layer += [nn.Conv2d(ngf * mult, ngf * mult, kernel_size=1, stride=1, padding=0, bias=False),
                        ILN(int(ngf * mult)),
                        nn.ReLU(True)]
        Decode_mask = []
        for i in range(n_downsampling):
            # Decode_mask += [nn.Upsample(scale_factor=2, mode='nearest')]
            Decode_mask += [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)]
        # Decode_mask = []
        # for i in range(n_downsampling):
        #     mult = 2 ** (n_downsampling - i)
        #     Decode_mask += [nn.Upsample(scale_factor=2, mode='nearest'),
        #                      nn.ReflectionPad2d(1),
        #                      nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False),
        #                      ILN(1),
        #                      nn.ReLU(True)]
        #
        # Decode_mask += [nn.ReflectionPad2d(1),
        #                  nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False),
        #                  nn.Tanh()]

        mult = 4
        self.DownBlock = nn.Sequential(*DownBlock)
        self.FC = nn.Sequential(*FC)
        self.Decode_image = nn.Sequential(*Decode_image)

        # self.recon_middle = nn.Sequential(*recon_middle)
        self.Decode_recon = nn.Sequential(*Decode_recon)
        self.Decode_mask = nn.Sequential(*Decode_mask)

        self.Split_layer = nn.Sequential(*Split_layer)
        self.background_mask = torch.ones(1, ngf * mult, 64, 64, device='cuda')
        self.background_layer = torch.ones(1, 1, 64, 64, device='cuda')
        self.gamma_beta_one = torch.ones(1, ngf * mult, device='cuda')
        self.conv2x1 = nn.Sequential(*conv2x1)
        self.background_image = torch.ones(1, 3, 256, 256, device='cuda')

    def forward(self, input):
        x = self.DownBlock(input)
        x_r = x.clone()
        # x_decoder = x.clone()
        x = self.Split_layer(x)
        # cam分支
        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)  # [C,248,64,64]
        x = self.relu(self.conv1x1(x))  # [C,248,64,64]
        x_decoder = x.clone()

        # for i in range(0, x.size()[0]):
        #     for j in range(0, x.size()[1]):
        #         max_value, _ = torch.max(torch.max(x[i][j], 0)[0], 0)
        #         layer_value = max_value*self.background_layer - x[i:i+1, j:j+1, :, :]
        #         if(j==0):
        #             x_Down = layer_value
        #         else:
        #             x_Down = torch.cat([x_Down, layer_value], 1)
        heatmap = torch.sum(x, dim=1, keepdim=True)
        x_mask = x.clone()  # [C,248,64,64]
        # x_mask = heatmap.clone()  # [C,1,64,64]
        # x_mask = x_mask - torch.min(torch.min(x_mask[0][0], 0)[0], 0)[0]
        # att_mask = (x_mask/torch.max(torch.max(x_mask[0][0], 0)[0], 0)[0]-0.3)*10000
        # att_mask = nn.Sigmoid()(att_mask)
        att_mask = self.conv2x1(x_mask)
        # att_mask = att_mask - torch.min(torch.min(att_mask[0][0], 0)[0], 0)[0]
        # att_mask = (att_mask / torch.max(torch.max(att_mask[0][0], 0)[0], 0)[0] - 0.5) * 10000
        att_mask = nn.Sigmoid()(att_mask)
        att_mask = att_mask.repeat(1, self.ngf * 4, 1, 1)
        # x_mask = x_mask.repeat(1, self.ngf*4, 1, 1)
        # cam分支

        # AdaLIN三个分支
        if self.light:
            x_ = torch.nn.functional.adaptive_avg_pool2d(x, 1)
            x_ = self.FC(x_.view(x_.shape[0], -1))
            # x_Down_ = torch.nn.functional.adaptive_avg_pool2d(x_Down, 1)
            # x_Down_ = self.FC(x_Down_.view(x_Down_.shape[0], -1))
        else:
            x_ = self.FC(x.view(x.shape[0], -1))
            # x_Down_ = self.FC(x_Down.view(x_Down.shape[0], -1))
        gamma1, beta1 = self.gamma1(x_), self.beta1(x_)  # gamma1[C, 248] beta1 [C, 248]
        # gamma2, beta2 = self.gamma2(x_Down_), self.beta2(x_Down_)

        for i in range(self.n_blocks):
            # att_mask, x_mask = getattr(self, 'Mask_Layer_' + str(i+1))(x_mask)
            # att_mask = att_mask.repeat(1, self.ngf*4, 1, 1)
            # x_mask = x_mask.repeat(1, self.ngf*4, 1, 1)
            x_r = getattr(self, 'recon_middle_' + str(i + 1))(x_r)
            x_decoder1 = getattr(self, 'UpBlock1_' + str(i + 1))(x_decoder, gamma1, beta1)  # x_decoder[C,248,64,64]
            # x_decoder2 = getattr(self, 'UpBlock2_' + str(i+1))(x_r, gamma2, beta2)  # x_decoder[C,248,64,64]
            # x_decoder = x_decoder1*att_mask + x_decoder2*(self.background_mask-att_mask)
            # x_r = x_decoder2
            x_decoder = x_decoder1 * att_mask + x_r * (self.background_mask - att_mask)

            # Up_feature = torch.cat([x_decoder1, att_mask], 1)
            # Down_feature = torch.cat([x_r, self.background_mask-att_mask], 1)
            # x_decoder = torch.cat([x_decoder1*att_mask, x_r*(self.background_mask-att_mask)], 1)
            # x_decoder = getattr(self, 'Mask_Layer_' + str(i+1))(x_decoder)
            if (i == self.n_blocks - 1):
                image_out1 = self.Decode_image(x_decoder)
                input_r = self.Decode_recon(x_r)

                out_mask = self.Decode_mask(att_mask[:, 0:1, :, :])
                out_mask = out_mask.repeat(1, 3, 1, 1)
                image_out2 = image_out1 * out_mask + input * (self.background_image - out_mask)
        # AdaLIN三个分支

        return image_out1, cam_logit, heatmap, out_mask, image_out2, input_r


class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Recon_Middle(nn.Module):
    def __init__(self, dim, use_bias):
        super(Recon_Middle, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = ILN(dim)
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = ILN(dim)
    def forward(self, x):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.relu1(out)

        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out)
        return x + out

class ResnetAdaILNBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x

class Mask_Layer(nn.Module):
    def __init__(self, dim, use_bias):
        super(Mask_Layer, self).__init__()
        self.conv1 = nn.Conv2d(dim, int(dim/2), kernel_size=1, stride=1, padding=0, bias=use_bias)
        self.relu1 = nn.ReLU(True)
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)

        return out


class adaILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.9)

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)

        return out


class ILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)

        return out


class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        model = [nn.ReflectionPad2d(1),
                 nn.utils.spectral_norm(
                 nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
                 nn.LeakyReLU(0.2, True)]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True)]

        mult = 2 ** (n_layers - 2 - 1)
        model += [nn.ReflectionPad2d(1),
                  nn.utils.spectral_norm(
                  nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.gap_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.gmp_fc = nn.utils.spectral_norm(nn.Linear(ndf * mult, 1, bias=False))
        self.conv1x1 = nn.Conv2d(ndf * mult * 2, ndf * mult, kernel_size=1, stride=1, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

        self.pad = nn.ReflectionPad2d(1)
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(ndf * mult, 1, kernel_size=4, stride=1, padding=0, bias=False))

        self.model = nn.Sequential(*model)

    def forward(self, input):
        x = self.model(input)

        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        x = torch.cat([gap, gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))

        heatmap = torch.sum(x, dim=1, keepdim=True)

        x = self.pad(x)
        out = self.conv(x)

        return out, cam_logit, heatmap


class RhoClipper(object):

    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):

        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w
