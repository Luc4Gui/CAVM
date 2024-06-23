import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
import torch
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_log(logger,message):
    print(message, flush=True)
    if logger:
        logger.write(str(message) + '\n')
if __name__ == '__main__':
    set_seed(42)

    opt = TrainOptions().parse()
    #Training data
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)
    ##logger ##
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    logger = open(os.path.join(save_dir, 'log.txt'), 'w+')
    print_log(logger,opt.name)
    logger.close()
    #validation data
    opt.phase='val'
    data_loader_val = CreateDataLoader(opt)
    dataset_val = data_loader_val.load_data()
    dataset_size_val = len(data_loader_val)
    print('#Validation images = %d' % dataset_size)
    model = create_model(opt)
    
    opt.checkpoints_dir = '/root/checkpoints_1/'
    visualizer_1 = Visualizer(opt)
    opt.checkpoints_dir = '/root/checkpoints_2/'
    visualizer_2 = Visualizer(opt)
    opt.checkpoints_dir = '/root/checkpoints/'
    visualizer_0 = Visualizer(opt)
    visualizers = [visualizer_0,visualizer_1,visualizer_2]

    opt.checkpoints_dir = '/root/checkpoints_test/1/'
    visualizer_1_test = Visualizer(opt)
    opt.checkpoints_dir = '/root/checkpoints_test/2/'
    visualizer_2_test = Visualizer(opt)
    opt.checkpoints_dir = '/root/checkpoints_test/0/'
    visualizer_0_test = Visualizer(opt)
    visualizers_tets = [visualizer_0_test,visualizer_1_test,visualizer_2_test]


    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        #Training step
        opt.phase='train'
        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            for vi in range(3):
                visualizers[vi].reset()
                visualizers_tets[vi].reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                temp_visuals=model.get_current_visuals()
                for vi in range(3):
                    visualizers[vi].display_current_results(temp_visuals[vi], epoch, save_result)                    
                    
            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                for vi in range(3):
                    visualizers[vi].print_current_errors(epoch, epoch_iter, errors, t, t_data)
                if opt.display_id > 0:
                    for vi in range(3):
                        visualizers[vi].plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

            iter_data_time = time.time()
        #Validaiton step
        if epoch % opt.save_epoch_freq == 0:
            L1_avg=[]
            psnr_avg=[]
            logger = open(os.path.join(save_dir, 'log.txt'), 'a')
            print(opt.dataset_mode)
            opt.phase='val'
            for i, data_val in enumerate(dataset_val):
#        		    
                model.set_input(data_val)
#        		    
                model.test()
#        		    
                fake_ims = [fake_B.cpu().data.numpy()/2 for fake_B in model.fake_B] 
#        		    
                real_ims= [real_B.cpu().data.numpy()/2 for real_B in model.inputs_B]
#        		    
                real_ims = [np.clip(real_im,0,1) for real_im in real_ims]
#        		    
                fake_ims = [np.clip(fake_im,0,1) for fake_im in fake_ims]

                L1_avg.append([abs(fake_im-real_im).mean() for fake_im,real_im in zip(fake_ims,real_ims)])
                psnr_avg.append([psnr(fake_im, real_im, data_range=1) for fake_im,real_im in zip(fake_ims,real_ims)])

                if i % 10 == 0:
                    save_result = i % 10 == 0
                    temp_visuals=model.get_current_visuals()
                    for vi in range(3):
                        visualizers_tets[vi].display_current_results(temp_visuals[vi], i, save_result)                    
#                 
            l1_avg_loss = [np.mean(np.array(L1_avg)[:,lo]) for lo in range(3)]
#                
            mean_psnr = [np.mean(np.array(psnr_avg)[:,lo]) for lo in range(3)]
#                
            std_psnr = [np.std(np.array(psnr_avg)[:,lo]) for lo in range(3)]
#                
            print_log(logger,'Epoch %3d   l1_avg_loss0: %.5f   l1_avg_loss1: %.5f    l1_avg_loss2: %.5f   mean_psnr_0: %.3f  mean_psnr_1: %.3f    mean_psnr_2: %.3f  std_psnr_0:%.3f  std_psnr_1:%.3f  std_psnr_2:%.3f  ' % \
            (epoch, l1_avg_loss[0], l1_avg_loss[1], l1_avg_loss[2], mean_psnr[0] , mean_psnr[1], mean_psnr[2], std_psnr[0], std_psnr[1], std_psnr[2]))
# #               
            print_log(logger,'')
            logger.close()
    		   #
            print('saving the model at the end of epoch %d, iters %d' %(epoch, total_steps))
#        		    
            model.save('latest')
#        		   
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
