###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.npy', '.pkl'
]

def is_A_file_pretrain(filename):
    return any(filename.endswith(extension) for extension in ['t1n.nii.gz'])

def is_Mpre_file(filename):
    return any(filename.endswith(extension) for extension in ['seg-pre.nii.gz'])

def is_M_real_file(filename):
    return any(filename.endswith(extension) for extension in ['seg-gt.nii.gz'])

def is_B_file_pre_o(filename):
    return any(filename.endswith(extension) for extension in ['t1c-033.nii.gz', 't1c-066.nii.gz', 't1c.nii.gz'])

def is_B_file_pre_in(filename):
    return any(filename.endswith(extension) for extension in ['t1c-033.nii.gz', 't1c-066.nii.gz'])

def is_A0_file_pretrain(filename):
    return any(filename.endswith(extension) for extension in ['t1n.nii.gz', 't2f.nii.gz', 't2w.nii.gz'])


def make_dataset_nii_pretrain_A(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, dirs, _ in sorted(os.walk(dir)):
        for dir in dirs:
            images_per = []
            for file in os.listdir(os.path.join(root,dir)):
                if is_A_file_pretrain(file):
                    path = os.path.join(root, dir, file)
                    images_per.append(path)
            
            images.append(images_per)

    return images

def make_dataset_nii_pretrain_M_real(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, dirs, _ in sorted(os.walk(dir)):
        for dir in dirs:
            images_per = []
            for file in os.listdir(os.path.join(root,dir)):
                if is_M_real_file(file):
                    path = os.path.join(root, dir, file)
                    images_per.append(path)
            
            images.append(images_per)

    return images

def make_dataset_nii_pretrain_Mpre(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, dirs, _ in sorted(os.walk(dir)):
        for dir in dirs:
            images_per = []
            for file in os.listdir(os.path.join(root,dir)):
                if is_Mpre_file(file):
                    path = os.path.join(root, dir, file)
                    images_per.append(path)
            
            images.append(images_per)

    return images

def make_dataset_nii_pretrain_B_o(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, dirs, _ in sorted(os.walk(dir)):
        for dir in dirs:
            images_per = []
            for file in os.listdir(os.path.join(root,dir)):
                if is_B_file_pre_o(file):
                    path = os.path.join(root, dir, file)
                    images_per.append(path)
            print(images_per)
            images.append(images_per) # n * 3 (02 10 true)

    return images


def make_dataset_nii_pretrain_B_in(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, dirs, _ in sorted(os.walk(dir)):
        for dir in dirs:
            images_per = []
            for file in os.listdir(os.path.join(root,dir)):
                if is_B_file_pre_in(file):
                    path = os.path.join(root, dir, file)
                    images_per.append(path)
            images.append(images_per) # n * 3 (02 10 true)

    return images


def make_dataset_nii_pretrain_A0(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, dirs, _ in sorted(os.walk(dir)):
        for dir in dirs:
            images_per = []
            for file in os.listdir(os.path.join(root,dir)):
                if is_A0_file_pretrain(file):
                    path = os.path.join(root, dir, file)
                    images_per.append(path)
            
            images.append(images_per)

    return images
