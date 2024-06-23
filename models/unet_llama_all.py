import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torchvision import models
from torch.optim.lr_scheduler import LambdaLR

class UNetLlama_model(BaseModel):
    def name(self):
        return 'UNetLlama_model'

    def lr_lambda(self, step):
        step = step + 1
        d_model = 384
        warmup_steps = 3000
        return d_model**(-0.5) * min(step**(-0.5), step * warmup_steps**(-1.5))

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        
        self.which_model_netG = opt.which_model_netG
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG,opt.vit_name,opt.fineSize,opt.pre_trained_path, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                      pre_trained_trans=opt.pre_trained_transformer,pre_trained_resnet = opt.pre_trained_resnet)

        self.netG.load_state_dict(torch.load('/root/checkpoints/ASNR_swinunetr_preunet_ssim_4_128_true/25_net_G.pth'), strict=False)

        for param in self.netG.parameters():
            param.requires_grad = False
        for param in self.netG.model_llamalike4.parameters():
            param.requires_grad = True
        for param in self.netG.model_llamalike3.parameters():
            param.requires_grad = True

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)

        if self.isTrain:
            self.criterionL2 = torch.nn.MSELoss()
            self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.96))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')

    def set_input(self, inputs):

        self.inputs_A = [inputs['A'][:,:,i].cuda() for i in range(3)]
        self.inputs_B = [inputs['B'][:,:,i].cuda() for i in range(3)]

        self.inputs_A[0] = inputs['A0'].cuda()

        self.image_paths = inputs['A_paths']

    def forward(self):\
        self.fake_f_3, self.fake_f_4, self.fake_B = self.netG.test(self.inputs_A[0], self.inputs_A[0])
    
    def test(self):
        with torch.no_grad():
            self.fake_f_3, self.fake_f_4, self.fake_B = self.netG.test(self.inputs_A[0], self.inputs_A[0])
            
    def test_latent(self):
        with torch.no_grad():
            self.fake_f_3, self.fake_f_4 = self.netG.test_latent(self.inputs_A[0], self.inputs_A[0])
        
    # get image paths
    def get_image_paths(self):
        return self.image_paths
        
    def backward_G(self):
        self.loss_G_L2 = self.criterionL2(self.fake_B[0], self.inputs_B[0]) * self.opt.lambda_A

        for index in range(1,3):
            self.loss_G_L2 += self.criterionL2(self.fake_B[index], self.inputs_B[index]) * self.opt.lambda_A

        self.loss_G_L2.backward()
    
    def optimize_parameters(self):
        self.forward()
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([
                            ('G_L2', self.loss_G_L2.item())
                            ])
    
    def get_current_visuals(self):
        real_As = [util.tensor2im_01(real_A.data) for real_A in self.inputs_A]
        fake_Bs = [util.tensor2im_01(fake_B.data) for fake_B in self.fake_B]
        real_Bs = [util.tensor2im_01(real_B.data) for real_B in self.inputs_B]
        return [OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)]) for real_A, real_B, fake_B in zip(real_As,real_Bs,fake_Bs)]

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
