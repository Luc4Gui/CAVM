python3 train_llama.py --dataroot /root/ASNR_3D/ --name ASNR_swinunetr_llama-fine-all-128-4-tr-image --gpu_ids 0 --model unet_llama_all --which_model_netD 3d \
    --which_model_netG swin_unetr_llama_3 --which_direction AtoB --lambda_A 10 --dataset_mode nii_3D_fine --norm batch --beta1 0.9 --pool_size 0 --batchSize 1\
    --output_nc 1 --input_nc 1 --loadSize 192 --fineSize 192 --niter 25 --niter_decay 25 --save_epoch_freq 1 --nThreads 32\
    --serial_batches --no_flip --checkpoints_dir /root/checkpoints/ --display_id 0 --lr 0.0001
