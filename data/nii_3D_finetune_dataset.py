import os.path
from data.base_dataset import BaseDataset, custom_transform_pre
from data.image_folder import make_dataset_nii_pretrain_A,  make_dataset_nii_pretrain_Mpre, make_dataset_nii_pretrain_M_real, make_dataset_nii_pretrain_B_o, make_dataset_nii_pretrain_B_in, make_dataset_nii_pretrain_A0
from PIL import Image
import random
import numpy as np
import torch
import nibabel as  nb

class Nii3DPREDataset(BaseDataset):
    def initialize(self, opt = ''):
        self.root = '/root/ASNR_3D'
        if opt == '':
            self.dir_A = os.path.join(self.root,'train' + 'A')
            self.dir_B = os.path.join(self.root, 'train' + 'B')
        else:
            self.dir_A = os.path.join(self.root, opt.phase + 'A')
            self.dir_B = os.path.join(self.root, opt.phase + 'B')
        self.A_paths = make_dataset_nii_pretrain_A(self.dir_A)
        self.A0_paths = make_dataset_nii_pretrain_A0(self.dir_A)
        self.Mpre_paths = make_dataset_nii_pretrain_Mpre(self.dir_A)
        self.M_real_paths = make_dataset_nii_pretrain_M_real(self.dir_A)
        self.B_paths = make_dataset_nii_pretrain_B_o(self.dir_B)
        self.A_paths_add = make_dataset_nii_pretrain_B_in(self.dir_B)

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)

    def __getitem__(self, index):
        A_paths = self.A_paths[index % self.A_size]
        A0_paths = self.A0_paths[index % self.A_size]
        A_paths_add = self.A_paths_add[index % self.A_size]
        B_paths = self.B_paths[index % self.B_size]
        Mpre_paths = self.Mpre_paths[index % self.A_size]
        M_real_paths = self.M_real_paths[index % self.A_size]

        A_imgs = torch.from_numpy(np.array([np.asanyarray(nb.load(A_path).dataobj) for A_path in A_paths] + [np.asanyarray(nb.load(A_path_add).dataobj) for A_path_add in A_paths_add])).float().unsqueeze(0)
        B_imgs = torch.from_numpy(np.array([np.asanyarray(nb.load(B_paths[2]).dataobj)]+[np.asanyarray(nb.load(B_paths[2]).dataobj)]+[np.asanyarray(nb.load(B_paths[2]).dataobj)])).float().unsqueeze(0)
        M_imgs = torch.from_numpy(np.array([np.asanyarray(nb.load(M_path).dataobj)for M_path in Mpre_paths])).float().unsqueeze(0)
        M_resl_imgs = torch.from_numpy(np.array([np.asanyarray(nb.load(M_path).dataobj)for M_path in M_real_paths])).float().unsqueeze(0)
        A0_imgs = torch.from_numpy(np.array([np.asanyarray(nb.load(A0_path).dataobj) for A0_path in A0_paths] + [np.asanyarray(nb.load(M_path).dataobj)for M_path in Mpre_paths])).float().unsqueeze(0)
        
        A = custom_transform_pre(A_imgs)
        B = custom_transform_pre(B_imgs)
        A0 = custom_transform_pre(A0_imgs)
        M = custom_transform_pre(M_imgs)
        M_real = custom_transform_pre(M_resl_imgs)

        return {'A': A, 'B': B, 'M': M[0,0], 'M_real': M_real[0,0], 'A0': A0[0],
                'A_paths': A_paths[0], 'B_paths': B_paths[0]}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'Nii3DPREDataset'
