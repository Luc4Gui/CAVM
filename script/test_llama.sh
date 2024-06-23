
python3 test_llama.py --dataroot /root/ASNR_3D/ --name ASNR_swinunetr_llama-fine-all-128-4-tr-image --gpu_ids 0 --model unet_llama_all --which_model_netD 3d \
    --which_model_netG swin_unetr_llama_3 --which_direction AtoB --dataset_mode nii_3D_fine --norm batch --batchSize 1\
    --how_many 10000 --output_nc 1 --input_nc 1 --loadSize 192 --fineSize 192 --nThreads 32\
    --serial_batches --no_flip --results_dir /root/results_att/ --checkpoints_dir /root/checkpoints/ --which_epoch latest
