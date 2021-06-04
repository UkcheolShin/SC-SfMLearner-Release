import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random
import os
import cv2
import scipy.ndimage as ndi

def load_as_float(path):
    return imread(path).astype(np.float32)


class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .
        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, train=True, sequence_length=3, transform=None, skip_frames=1, dataset='kitti', mod = None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.transform = transform
        self.dataset = dataset
        self.k = skip_frames
        self.crawl_folders(sequence_length)

        if mod == 'box_blur':
            self.img_mod = self.box_blur
        elif mod == 'gauss_blur':
            self.img_mod = self.box_blur
        elif mod == 'motion_blur':
            self.img_mod = self.motion_blur
        elif mod == 'bilateral_blur':
            self.img_mod = self.bilateral_blur
        elif mod == 'sharpening':
            self.img_mod = self.sharpening
        elif mod == 'hist_eq':
            self.img_mod = self.hist_eq
        elif mod == 'rgb2gray':
            self.img_mod = self.rgb2gray
        elif mod == None :
            self.img_mod = self.no_mod

        # import pdb
        # pdb.set_trace()
#         sample = self.samples[1]
#         tgt_img = load_as_float(sample['tgt'])

#         import matplotlib.pyplot as plt
# #        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         plt.imshow(tgt_img.astype(np.uint8))
#         plt.show()
#         abc1 = self.box_blur(tgt_img)
#         plt.imshow(abc1.astype(np.uint8))
#         plt.show()
#         abc2 = self.gauss_blur(tgt_img)
#         plt.imshow(abc2.astype(np.uint8))
#         plt.show()
#         abc3 = self.motion_blur(tgt_img,5)
#         plt.imshow(abc3.astype(np.uint8))
#         plt.show()
#         abc4 = self.bilateral_blur(tgt_img, 7)
#         plt.imshow(abc4.astype(np.uint8))
#         plt.show()
#         abc5 = self.sharpening(tgt_img.astype(np.uint8))
#         plt.imshow(abc5.astype(np.uint8))
#         plt.show()
#         abc6 = self.hist_eq(tgt_img)
#         plt.imshow(abc6.astype(np.uint8))
#         plt.show()

    def crawl_folders(self, sequence_length):
        # k skip frames
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length * self.k, demi_length * self.k + 1, self.k))
        shifts.pop(demi_length)
        for scene in self.scenes:
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            imgs = sorted(scene.files('*.jpg'))

            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length * self.k, len(imgs)-demi_length * self.k):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                sequence_set.append(sample)
        random.shuffle(sequence_set)
        self.samples = sequence_set

    def box_blur(self, img, kernel_size = 11) : 
        # box blur
#       kernel_size = 3 # 3,5,7,...
        outimg = cv2.blur(img, ksize=(kernel_size, kernel_size))
        return outimg

    def gauss_blur(self, img, sigma = 0.5) : 
        # gaussian blur
#        sigma = 0.5 # 0.15~ 3
        kernel_size = 2*int(4*sigma + 0.5) + 1
        outimg = cv2.GaussianBlur(img, ksize=(kernel_size,kernel_size), sigmaX=sigma)
        return outimg

    def motion_blur(self, img, kernel_size=5) : 
        # motion blur
#        kernel_size = 3
        angle = random.randint(0, 359)
        direction = random.uniform(0.,1.) # 1 : forward, 0.5: uniform, 0 : backward
        motion_blur = np.zeros((kernel_size, kernel_size))
        direct = [direction + (1-2*direction) / (kernel_size - 1) * i for i in range(kernel_size)]
        motion_blur[int((kernel_size-1)/2), :] = np.array(direct)
        motion_blur = ndi.rotate(motion_blur, angle, reshape=False)
        motion_blur = motion_blur / motion_blur.sum()
        outimg = cv2.filter2D(img, -1, motion_blur)
        return outimg

    def bilateral_blur(self, img, d_size=15) : 
        # bilateral blur
        outimg = cv2.bilateralFilter(img, d=d_size, sigmaColor=75, sigmaSpace=75)
        return outimg

    def rgb2gray(self, img) : 
        # rgb2gray
#        outimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#        return np.expand_dims(outimg,2)
#        outimg = np.expand_dims(img.mean(2), 2)
        outimg = (img - 100.0).clip(0.,255.)
        return outimg

    def sharpening(self, img) : 
        # sharpening
        #sharpening_1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpening_1 = np.array([[-1, -1, -1, -1, -1], \
                                [-1, 2, 2, 2, -1], \
                                [-1, 2, 9, 2, -1],\
                                [-1, 2, 2, 2, -1],\
                                [-1, -1, -1, -1, -1]]) / 9.0
        outimg = cv2.filter2D(img.astype(np.uint8), -1, sharpening_1)
        return outimg.astype(np.float32)

    def hist_eq(self, img) : 
        # hist eq
        img_yuv = cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_RGB2YUV) # Y : intensity, u,v : color
        img_y = cv2.equalizeHist(img_yuv[:,:,0])
        img_yuv[:,:,0] = img_y
        outimg = cv2.cvtColor(img_yuv,cv2.COLOR_YUV2RGB) # Y : intensity, u,v : color
        return outimg.astype(np.float32)

    def no_mod(self, img) : 
        return img

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        tgt_img_mod  = self.img_mod(tgt_img)
        ref_imgs_mod = []
        for img in ref_imgs : ref_imgs_mod.append(self.img_mod(img))

        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs + [tgt_img_mod] + ref_imgs_mod, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:3]
            tgt_img_mod = imgs[3]
            ref_imgs_mod = imgs[4:6]
        else:
            intrinsics = np.copy(sample['intrinsics'])
        return tgt_img, ref_imgs, tgt_img_mod, ref_imgs_mod, intrinsics, np.linalg.inv(intrinsics)

    def __len__(self):
        return len(self.samples)
