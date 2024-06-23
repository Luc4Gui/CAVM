def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'unet_llama_all':
        from .unet_llama_all import UNetLlama_model
        model = UNetLlama_model()
    elif opt.model == 'unet_backbone_ssim_3':
        from .unet_backbone_ssim_3 import UNetbackbone_model
        model = UNetbackbone_model()
    else:
        raise NotImplementedError('model [%s] not implemented.' % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
