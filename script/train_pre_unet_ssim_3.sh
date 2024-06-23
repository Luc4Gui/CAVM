python3 train_pre_unet_3.py --dataroot /root/ASNR_3D/ --name ASNR_swinunetr_preunet_ssim_4_128_true --gpu_ids 0 --model unet_backbone_ssim_3 --which_model_netD 3d \
    --which_model_netG swin_unetr_backbone_3 --which_direction AtoB --lambda_A 100 --dataset_mode nii_3D_pre_train_3 --norm batch --pool_size 0 --batchSize 1\
    --output_nc 1 --input_nc 1 --loadSize 192 --fineSize 192 --niter 25 --niter_decay 25 --save_epoch_freq 5 --nThreads 32 --beta1 0.9 \
    --serial_batches --no_flip --checkpoints_dir /root/checkpoints/ --display_id 0 --lr 0.0001
