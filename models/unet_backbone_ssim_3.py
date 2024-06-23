import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torchvision import models
from torch.optim.lr_scheduler import LambdaLR

class UNetbackbone_model(BaseModel):
    def name(self):
        return 'UNetbackbone_model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        # load/define networks
        self.which_model_netG = opt.which_model_netG
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG,opt.vit_name,opt.fineSize,opt.pre_trained_path, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                      pre_trained_trans=opt.pre_trained_transformer,pre_trained_resnet = opt.pre_trained_resnet)

        self.fake_B = [0,0,0,0]
        self.x_embedding = [0,0,0,0]
        self.t1_embedding = [0,0,0,0]

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)

        if self.isTrain:
            self.lambda_f = opt.lambda_f
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,opt.vit_name,opt.fineSize,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.fake_AB_pool = ImagePool(opt.pool_size)
            
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, inputs):

        self.inputs_A = [inputs['A'][:,:,i].cuda() for i in range(4)]
        self.input_A0 = inputs['A0'].cuda()
        self.inputs_B = [inputs['B'][:,:,i].cuda() for i in range(4)]

        self.image_paths = inputs['A_paths']

    def forward(self, index):
        if index ==0 :
            self.fake_B[index] = self.netG.forward_0(self.inputs_A[index], self.input_A0)
        else:
            self.fake_B[index] = self.netG(self.inputs_A[index], self.input_A0)
        
    # no backprop gradients
    def test(self):
        with torch.no_grad():
            for index in range(0, len(self.inputs_A)):
                if index ==0 :
                    self.fake_B[index] = self.netG.forward_0(self.inputs_A[index], self.input_A0)
                else:
                    self.fake_B[index] = self.netG(self.inputs_A[index], self.input_A0)

    # get image paths
    def get_image_paths(self):
        return self.image_paths
    
    def backward_D(self, index):
        if index > 0:
            index_ = index-1
        else:
            index_ = index
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.inputs_A[index_], self.fake_B[index]), 1).data)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False) #
        # Real
        real_AB = torch.cat((self.inputs_A[index_], self.inputs_B[index]), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5*self.opt.lambda_adv

        self.loss_D.backward()
        
    def backward_G(self, index):
        if index > 0:
            index_ = index-1
        else:
            index_ = index
        fake_AB = torch.cat((self.inputs_A[index_], self.fake_B[index]), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)*self.opt.lambda_adv
        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B[index], self.inputs_B[index]) * self.opt.lambda_A
        
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        # self.loss_G = self.loss_G_L2
        
        self.loss_G.backward()

    def optimize_parameters(self):
        for index in range(0, len(self.inputs_A)):
            self.forward(index)

            self.optimizer_D.zero_grad()
            self.backward_D(index)
            self.optimizer_D.step()
            
            self.optimizer_G.zero_grad()
            self.backward_G(index)
            self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.item()),
                            ('G_L1', self.loss_G_L1.item()),
                            ('D_real', self.loss_D_real.item()),
                            ('D_fake', self.loss_D_fake.item())
                            ])

    def get_current_visuals(self):
        real_As = [util.tensor2im_01(self.inputs_A[i].data) for i in range(4)]
        fake_Bs = [util.tensor2im_01(self.fake_B[i].data) for i in range(4)]
        real_Bs = [util.tensor2im_01(self.inputs_B[i].data) for i in range(4)]
        return [OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)]) for real_A, real_B, fake_B in zip(real_As,real_Bs,fake_Bs)]

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
