import os
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from util import html
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as compare_ssim
from collections import OrderedDict
import torch
import torch.nn.functional as F
from evalue.evaluation_utils import compute_metrics_mask, compute_metrics

def dice_coefficient(input, target, threshold=0.5):
    smooth = 1e-6  
    input_bin = (input > threshold).float()
    target_bin = (target > 0.1).float()
    
    iflat = input_bin.view(-1)
    tflat = target_bin.view(-1)
    intersection = (iflat * tflat).sum()
    
    return (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


def numpy2im(image_3D, imtype=np.uint8):
    slice_num = image_3D.shape[-1]
    image_numpy = image_3D[:,:,slice_num//2]
    image_numpy = np.repeat(image_numpy[:, :, np.newaxis], 3, axis=2)
#     print(image_numpy.shape)
    image_numpy = image_numpy * 255.0
    return image_numpy.astype(imtype)

def calculate_similarity(x,y):
    x_flat = x.view(-1).unsqueeze(0)
    y_flat = y.view(-1).unsqueeze(0)

    similarity = F.cosine_similarity(x_flat, y_flat)
#     print(similarity)
    return similarity

def calculate_l1(x,y):
    x_flat = x.view(-1).unsqueeze(0)
    y_flat = y.view(-1).unsqueeze(0)

    similarity = F.l1_loss(x_flat, y_flat)
#     print(similarity)
    return similarity

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 32 
    opt.batchSize = 1  
    opt.serial_batches = True  
    opt.no_flip = True 
    print(opt.dataset_mode)

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()

    health_mse_list = [[],[],[]]
    health_psnr_list = [[],[],[]]
    health_ssim_list = [[],[],[]]

    tumor_mse_list = [[],[],[]]
    tumor_psnr_list = [[],[],[]]
    tumor_ssim_list = [[],[],[]]

    model = create_model(opt)
    opt.checkpoints_dir = '/root/checkpoints_1/'
    visualizer_1 = Visualizer(opt)
    opt.checkpoints_dir = '/root/checkpoints_2/'
    visualizer_2 = Visualizer(opt)
    opt.checkpoints_dir = '/root/checkpoints/'
    visualizer_0 = Visualizer(opt)
    visualizers = [visualizer_0,visualizer_1,visualizer_2]

    opt.results_dir = '/root/results_2/'
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage_2 = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    opt.results_dir = '/root/results_1/'
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage_1 = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    opt.results_dir = '/root/results/'
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage_0 = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    webpage = [webpage_0, webpage_1, webpage_2]

    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break

        model.set_input(data)
        model.test()
        
        real_mask = data['M_real'].unsqueeze(0).cuda()
        real_mask[real_mask>0] = 1

        for state in range(3):

            real_im_vis=np.clip(model.inputs_B[state][0,0,:,:,:].cpu().data.numpy()/2,0,1)
    #        		    
            fake_im_vis=np.clip(model.fake_B[state][0,0,:,:,:].cpu().data.numpy()/2,0,1)

            real_im_A_vis=np.clip(model.inputs_A[state][0,0,:,:,:].cpu().data.numpy()/2,0,1)
    #        		    
            fake_im=model.fake_B[state]
    #        		    
            real_im=model.inputs_B[state]

            real_im_A=model.inputs_A[state]

            health_mse, health_psnr, health_ssim = compute_metrics_mask(gt_image=real_im, prediction=fake_im, mask=(1-real_mask).bool())

            tumor_mse, tumor_psnr, tumor_ssim = compute_metrics_mask(gt_image=real_im, prediction=fake_im, mask=real_mask.bool())

            img_path = model.get_image_paths()
            print('%04d: process image... %s' % (i, img_path))

            print('health: ', health_mse, '\t', health_psnr, '\t', health_ssim)
            health_mse_list[state].append(health_mse)
            health_psnr_list[state].append(health_psnr)
            health_ssim_list[state].append(health_ssim)
            if torch.any(real_mask.bool()):
                print('tumor: ', tumor_mse, '\t', tumor_psnr, '\t', tumor_ssim)
                tumor_mse_list[state].append(tumor_mse)
                tumor_psnr_list[state].append(tumor_psnr)
                tumor_ssim_list[state].append(tumor_ssim)

            real_A_im = numpy2im(real_im_A_vis)
            real_B_im = numpy2im(real_im_vis)
            fake_B_im = numpy2im(fake_im_vis)
            visuals = OrderedDict([('real_A', real_A_im), ('fake_B', fake_B_im), ('real_B', real_B_im)])
            visualizers[state].save_images(webpage[state], visuals, img_path, aspect_ratio=opt.aspect_ratio)
        
    for state in range(3):

        mean_health_ssim = np.mean(np.array(health_ssim_list[state]))
        mean_health_psnr = np.mean(np.array(health_psnr_list[state]))
        mean_health_mse = np.mean(np.array(health_mse_list[state]))

        mean_tumor_ssim = np.mean(np.array(tumor_ssim_list[state]))
        mean_tumor_psnr = np.mean(np.array(tumor_psnr_list[state]))
        mean_tumor_mse = np.mean(np.array(tumor_mse_list[state]))

        std_health_ssim = np.std(np.array(health_ssim_list[state]))
        std_health_psnr = np.std(np.array(health_psnr_list[state]))
        std_health_mse = np.std(np.array(health_mse_list[state]))

        std_tumor_ssim = np.std(np.array(tumor_ssim_list[state]))
        std_tumor_psnr = np.std(np.array(tumor_psnr_list[state]))
        std_tumor_mse = np.std(np.array(tumor_mse_list[state]))

        print('state: ', state)
        print('mean_health_ssim: %.5f  std_health_ssim: %.5f  mean_tumor_ssim: %.5f  std_tumor_ssim: %.5f  ' % \
                (mean_health_ssim, std_health_ssim, mean_tumor_ssim, std_tumor_ssim))
        print('mean_health_psnr: %.5f  std_health_psnr: %.5f  mean_tumor_psnr: %.5f  std_tumor_psnr: %.5f  ' % \
                (mean_health_psnr, std_health_psnr, mean_tumor_psnr, std_tumor_psnr))
        print('mean_health_mse: %.5f  std_health_mse: %.5f  mean_tumor_mse: %.5f  std_tumor_mse: %.5f  ' % \
                (mean_health_mse, std_health_mse, mean_tumor_mse, std_tumor_mse))
        
        webpage[state].save()

        import datetime

        now = datetime.datetime.now()
        formatted_time = now.strftime('%Y-%m-%d_%H-%M-%S')
        filename = f'llama_test_log_{formatted_time}.txt'
        with open(filename, 'a') as file:
                file.write('mean_health_ssim: %.5f  std_health_ssim: %.5f  mean_tumor_ssim: %.5f  std_tumor_ssim: %.5f  ' % \
                        (mean_health_ssim, std_health_ssim, mean_tumor_ssim, std_tumor_ssim))
                file.write('mean_health_psnr: %.5f  std_health_psnr: %.5f  mean_tumor_psnr: %.5f  std_tumor_psnr: %.5f  ' % \
                        (mean_health_psnr, std_health_psnr, mean_tumor_psnr, std_tumor_psnr))
                file.write('mean_health_mse: %.5f  std_health_mse: %.5f  mean_tumor_mse: %.5f  std_tumor_mse: %.5f  ' % \
                        (mean_health_mse, std_health_mse, mean_tumor_mse, std_tumor_mse))
        