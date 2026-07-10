import os
from glob import glob
import cv2
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms
from torchvision.transforms import *
import torch
import math
from os.path import join
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
from PIL import Image, ImageOps
import random
import scipy
from scipy import io
import mat73
from skimage.color import rgb2lab,lab2rgb

def tensor_to_image(tensor):
    if len(tensor.shape)==4 : #batch 포함
        b,c,h,w = tensor.size()
        tensor = tensor[0]
        if c == 3:
            pass
        elif c == 1:
            tensor = tensor.repeat(3,1,1)
        output = np.transpose(np.array(tensor), (1,2,0))
        output = Image.fromarray(np.clip(output * 255.0, 0, 255.0).astype('uint8'))
    elif len(tensor.shape) == 3:
        c,h,w = tensor.size()
        if c == 3:
            pass
        elif c == 1:
            tensor = tensor.repeat(3, 1, 1)
        output = np.transpose(np.array(tensor), (1,2,0))
        output = Image.fromarray(np.clip(output * 255.0, 0, 255.0).astype('uint8'))
    elif len(tensor.shape) == 2:
        h,w = tensor.size()
        tensor = tensor.repeat(3,1,1)
        output = np.transpose(np.array(tensor), (1,2,0))
        output = Image.fromarray(np.clip(output * 255.0, 0, 255.0).astype('uint8'))
    return output
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg",".bmp"])
def is_mat_file(filename):
    return any(filename.endswith(extension) for extension in [".mat"])
def is_txt_file(filename):
    return any(filename.endswith(extension) for extension in [".txt"])
def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    # y, _, _ = img.split()
    return img
def load_img_gray(filepath):
    img = Image.open(filepath).convert('L')
    # y, _, _ = img.split()
    return img

def resize_img():
    return torchvision.transforms.Resize([256,256])

def get_patch(img_in, nir_in, patch_size, ix=-1, iy=-1):
    (ih, iw) = img_in.size
    ip = patch_size

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))
    nir_in = nir_in[:,ix:ix+ip,iy:iy+ip]
    # nir_in = nir_in.crop((iy, ix, iy + ip, ix + ip))

    # info_patch = {'ix': ix, 'iy': iy, 'ip': ip}

    return img_in, nir_in

def get_patch_2(img_in, img2_in, nir_in, nir_in2,patch_size, ix=-1, iy=-1):
    # print(img_in.size)
    # print(nir_in.shape)
    (ih, iw) = img_in.size
    ip = patch_size

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))
    #
    # img_in = img_in[:, ix:ix + ip, iy:iy + ip]
    img2_in = img2_in.crop((iy, ix, iy + ip, ix + ip))
    nir_in = nir_in[:,iy:iy+ip,ix:ix+ip]
    # nir_in = nir_in.crop((iy, ix, iy + ip, ix + ip))
    nir_in2 = nir_in2[:,iy:iy+ip,ix:ix+ip]

    # info_patch = {'ix': ix, 'iy': iy, 'ip': ip}

    return img_in, img2_in, nir_in, nir_in2

def get_patch_2_1(img_in, img2_in, nir_in, nir_in2,patch_size, ix=-1, iy=-1):
    # print(img_in.size)
    # print(nir_in.shape)
    (ih, iw) = img_in.size
    ip = patch_size

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))
    #
    # img_in = img_in[:, ix:ix + ip, iy:iy + ip]
    img2_in = img2_in.crop((iy, ix, iy + ip, ix + ip))
    nir_in = nir_in[:,ix:ix+ip,iy:iy+ip]
    # nir_in = nir_in.crop((iy, ix, iy + ip, ix + ip))
    nir_in2 = nir_in2[:,ix:ix+ip,iy:iy+ip]

    # info_patch = {'ix': ix, 'iy': iy, 'ip': ip}

    return img_in, img2_in, nir_in, nir_in2

def get_patch_2_hoon(img_in, img2_in,patch_size, ix=-1, iy=-1):
    # print(img_in.size)
    # print(nir_in.shape)
    (ih, iw) = img_in.size
    ip = patch_size

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))
    #
    # img_in = img_in[:, ix:ix + ip, iy:iy + ip]
    img2_in = img2_in.crop((iy, ix, iy + ip, ix + ip))

    # nir_in = nir_in.crop((iy, ix, iy + ip, ix + ip))


    # info_patch = {'ix': ix, 'iy': iy, 'ip': ip}

    return img_in, img2_in

def get_patch_3(img_in, img2_in, nir_in, patch_size, ix=-1, iy=-1):
    (ih, iw) = img_in.size
    ip = patch_size

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))
    #img2_in = img2_in.crop((iy, ix, iy + ip, ix + ip))
    img2_in = img2_in[:, ix:ix + ip, iy:iy + ip]
    nir_in = nir_in[:, ix:ix + ip, iy:iy + ip]
    # nir_in = nir_in.crop((iy, ix, iy + ip, ix + ip))

    # info_patch = {'ix': ix, 'iy': iy, 'ip': ip}

    return img_in, img2_in, nir_in

def get_patch_hoon(img_in, img2_in, nir_in, nir_in2,patch_size, ix=-1, iy=-1):
    c,w,h = img2_in.size()
    # print(img2_in.size())
    ip = patch_size

    if ix == -1:
        ix = random.randrange(0, w - ip + 1)
    if iy == -1:
        iy = random.randrange(0, h - ip + 1)

    img_in = img_in[:, ix:ix + ip, iy:iy + ip]
    img2_in = img2_in[:, ix:ix + ip, iy:iy + ip]
    nir_in = nir_in[:,ix:ix+ip,iy:iy+ip]
    nir_in2 = nir_in2[:,ix:ix+ip,iy:iy+ip]
    # print(img2_in.size())

    return img_in, img2_in, nir_in, nir_in2

def load_1ch(filepath):
    img = Image.open(filepath).convert('L')
    # y, _, _ = img.split()
    return img
def get_patch_two(img_in,img_in2,patch_size, ix=-1, iy=-1):
    c,w,h = img_in.size()
    # print(img_in.size())
    ip = patch_size

    if ix == -1:
        ix = random.randrange(0, w - ip + 1)
    if iy == -1:
        iy = random.randrange(0, h - ip + 1)

    img_in = img_in[:, ix:ix + ip, iy:iy + ip]
    img_in2 = img_in2[:, ix:ix + ip, iy:iy + ip]

    return img_in,img_in2

def get_patch_one(img_in,patch_size, ix=-1, iy=-1):
    # print(img_in.shape)
    c,w,h = img_in.size()
    # print(img_in.size())
    ip = patch_size

    if ix == -1:
        ix = random.randrange(0, w - ip + 1)
    if iy == -1:
        iy = random.randrange(0, h - ip + 1)

    img_in = img_in[:, ix:ix + ip, iy:iy + ip]

    return img_in

def get_patch_11(img_in,patch_size, ix=-1, iy=-1):
    (ih, iw) = img_in.size
    ip = patch_size

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))
    #

    # info_patch = {'ix': ix, 'iy': iy, 'ip': ip}

    return img_in

def get_patch_2tensor(img_in,wb_img_in, patch_size=256, ix=-1, iy=-1): # img_in : 이미지 파일 w,h | wb_img_in : 텐서 c,h,w
    c ,ih, iw = img_in.size()
    ip = patch_size
    # print('iw,ih:{},{}'.format(iw - ip + 1,ih - ip + 1))
    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
        #ix = iw - ip
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)
        #iy = 0

    img_in = img_in[:,iy:iy+ip,ix:ix+ip]
    # wb_img_in = wb_img_in.crop((iy, ix, iy + ip, ix + ip))
    wb_img_in = wb_img_in[:,iy:iy+ip,ix:ix+ip]
    # nir_in = nir_in.crop((iy, ix, iy + ip, ix + ip))
    return img_in,wb_img_in

def get_patch_2tensor_center(img_in,wb_img_in, patch_size=[256,256], ix=-1, iy=-1): # img_in : 이미지 파일 w,h | wb_img_in : 텐서 c,h,w
    c ,ih, iw = img_in.size()
    ctr_h,ctr_w = ih//2,iw//2
    ip1,ip2 = patch_size

    img_in = img_in[:,ctr_h-ip1//2:ctr_h+ip1//2,ctr_w-ip2//2:ctr_w+ip2//2]
    # wb_img_in = wb_img_in.crop((iy, ix, iy + ip, ix + ip))
    wb_img_in = wb_img_in[:,ctr_h-ip1//2:ctr_h+ip1//2,ctr_w-ip2//2:ctr_w+ip2//2]
    # nir_in = nir_in.crop((iy, ix, iy + ip, ix + ip))
    return img_in,wb_img_in


def get_patch_cattensor(concat_tensor, patch_size=256, ix=-1, iy=-1): # img_in : 이미지 파일 w,h | wb_img_in : 텐서 c,h,w
    c ,ih, iw = concat_tensor.size()
    ip = patch_size
    # print('iw,ih:{},{}'.format(iw - ip + 1,ih - ip + 1))
    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    img_in = concat_tensor[:,iy:iy+ip,ix:ix+ip]

    return img_in

def get_patch_2tensor_cube(img_in,wb_img_in, patch_size=256, ix=2050, iy=1050): # img_in : 이미지 파일 w,h | wb_img_in : 텐서 c,h,w
    c ,ih, iw = img_in.size()
    ip = patch_size
    # print('iw,ih:{},{}'.format(iw - ip + 1,ih - ip + 1))
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    # while not(ix + ip >= 604 and iy + ip >= 309) :
    #     ix = random.randrange(0, iw - ip + 1)
    #     iy = random.randrange(0, ih - ip + 1)

    img_in = img_in[:,iy:iy+ip,ix:ix+ip]
    # wb_img_in = wb_img_in.crop((iy, ix, iy + ip, ix + ip))
    wb_img_in = wb_img_in[:,iy:iy+ip,ix:ix+ip]
    # nir_in = nir_in.crop((iy, ix, iy + ip, ix + ip))
    return img_in,wb_img_in

def get_patch_256(img_in,wb_img_in, patch_size=256, ix=-1, iy=-1): # img_in : 이미지 파일 w,h | wb_img_in : 텐서 c,h,w
    (ih, iw) = img_in.size
    ip = patch_size

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))
    # wb_img_in = wb_img_in.crop((iy, ix, iy + ip, ix + ip))
    wb_img_in = wb_img_in[:,ix:ix+ip,iy:iy+ip]
    # nir_in = nir_in.crop((iy, ix, iy + ip, ix + ip))


    return img_in,wb_img_in

def get_patch_128(img_in,wb_img_in, patch_size=128, ix=-1, iy=-1):
    (ih, iw) = img_in.size
    ip = patch_size

    if ix == -1:
        ix = random.randrange(0, (iw - ip)/4 + 1)
        ix *= 4
    if iy == -1:
        iy = random.randrange(0, (ih - ip)/4 + 1)
        iy *= 4

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))
    # wb_img_in = wb_img_in.crop((iy, ix, iy + ip, ix + ip))
    wb_img_in = wb_img_in[:, ix:ix + ip, iy:iy + ip]
    # nir_in = nir_in.crop((iy, ix, iy + ip, ix + ip))


    return img_in,wb_img_in,ix,iy

def get_patch_64(img_in,wb_img_in, patch_size=64, ix=-1, iy=-1):
    (ih, iw) = img_in.size
    ip = patch_size

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))
    # wb_img_in = wb_img_in.crop((iy, ix, iy + ip, ix + ip))
    wb_img_in = wb_img_in[:, ix:ix + ip, iy:iy + ip]
    # nir_in = nir_in.crop((iy, ix, iy + ip, ix + ip))


    return img_in,wb_img_in

def get_patch_256_1(img_in, patch_size=256, ix=-1, iy=-1): # img_in : 이미지 파일 w,h | wb_img_in : 텐서 c,h,w
    (ih, iw) = img_in.size
    ip = patch_size

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))

    return img_in

def get_patch_128_1(img_in, patch_size=128, ix=-1, iy=-1):
    (ih, iw) = img_in.size
    ip = patch_size
    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)
    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))
    return img_in

def get_patch_64_1(img_in, patch_size=64, ix=-1, iy=-1):
    (ih, iw) = img_in.size
    ip = patch_size

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))

    return img_in

delta_theta = torch.FloatTensor(torch.randint(low=0, high=1000, size=(1, 1, 1, 1)) * 2 * 3.141 / 1000)
delta_pi = torch.FloatTensor(torch.randint(low=0, high=1000, size=(1, 1, 1, 1)) * 2 * 3.141 / 1000)
def vector_rotate_img(img,delta_theta,delta_pi):
    # red, green, blue = img[:, 0, :, :].unsqueeze(1), img[:, 1, :, :].unsqueeze(1), img[:, 2, :, :].unsqueeze(1)
    red, green, blue = img[0, :, :].unsqueeze(0), img[1, :, :].unsqueeze(0), img[ 2, :, :].unsqueeze(0)
    Red,Green,Blue = torch.zeros_like(img[0]),torch.zeros_like(img[0]),torch.zeros_like(img[0])

    r = torch.sqrt(red ** 2 + blue ** 2 + green ** 2)
    # for h in range(img.size(1)):
    #     for w in range(img.size(2)):
    #         if r[h,w] == 0:
    #             Red[h,w],Green[h,w],Blue[h,w] = 0,0,0
    #         else:

    theta = torch.acos(blue / r)
    pi = torch.atan(green / red)
    output = torch.cat([r, theta, pi], dim=0)

    theta_1 = output[1, :, :].unsqueeze(0) + torch.FloatTensor(delta_theta).repeat(1, output.size(1), output.size(2))
    pi_1 = output[2, :, :].unsqueeze(0) + torch.FloatTensor(delta_pi).repeat( 1, output.size(1), output.size(2))

    Red = r * torch.sin(theta_1) * torch.cos(pi_1)
    Green = r * torch.sin(theta_1) * torch.sin(pi_1)
    Blue = r * torch.cos(theta_1)
    output = torch.cat([Red, Green, Blue], dim=0)
    output_1 = torch.where(torch.isnan(output.float()),torch.tensor(0.0, dtype=torch.float32),output)
    return output_1
def vector_rotate_ill(illuminant,delta_theta,delta_pi):
    red,green,blue = illuminant[0].unsqueeze(0),illuminant[1].unsqueeze(0),illuminant[2].unsqueeze(0)
    r = torch.sqrt(red ** 2 + blue ** 2 + green ** 2)
    theta = torch.acos(blue / r)
    pi = torch.atan(green / red)
    output = torch.cat([r, theta, pi], dim=0)

    theta_1 = output[1] + delta_theta.squeeze()
    pi_1 = output[2] + delta_pi.squeeze()

    Red = r * torch.sin(theta_1) * torch.cos(pi_1)
    Green = r * torch.sin(theta_1) * torch.sin(pi_1)
    Blue = r * torch.cos(theta_1)

    return torch.cat([Red, Green, Blue], dim=0)

class DatasetFromFolder_our(data.Dataset):
    def __init__(self, image_dir, gt_ilu,folder, isTrain, transform=None):
        super(DatasetFromFolder_our, self).__init__()

        self.Train = isTrain
        self.istrain = folder

        self.im_dir = os.path.join(image_dir, self.istrain, 'rgb/')
        self.vis_dir = os.path.join(image_dir, self.istrain, 'estimated_vis_8ch/mat/')
        self.nir_dir = os.path.join(image_dir, self.istrain, 'nir_8ch/')

        self.image_filenames = [join(self.im_dir, x) for x in listdir(self.im_dir) if is_image_file(x)]
        self.vis_filenames = [join(self.vis_dir, x) for x in listdir(self.vis_dir) if is_mat_file(x)]
        self.nir_filenames = [join(self.nir_dir, x) for x in listdir(self.nir_dir) if is_mat_file(x)]

        self.transform = transform
        self.gt_ilu = gt_ilu
        #self.resize = resize_img()


    def __getitem__(self, index):

        input_im = load_img(self.image_filenames[index])
        #input_im = self.transform(input_im)
        #vis_input_im = mat73.loadmat(self.vis_filenames[index])
        vis_input_im = scipy.io.loadmat(self.vis_filenames[index])
        vis_input_im = vis_input_im['mat_data']
        vis_input_im = self.transform(vis_input_im)

        #C, H, W = vis_input_im.shape
        nir_input_im = mat73.loadmat(self.nir_filenames[index])
        #nir_input_im = scipy.io.loadmat(self.nir_filenames[index])
        nir_input_im = nir_input_im['mat_data']
        nir_input_im = self.transform(nir_input_im)
        gt = self.gt_ilu[index]
        gt_label = torch.from_numpy(gt)


        if self.Train == True:

            input_patch, vis_patch,nir_patch = get_patch_3(input_im, vis_input_im,nir_input_im,patch_size=256)

            if self.transform:
                input_patch = self.transform(input_patch)


            return input_patch.float(),vis_patch.float(),nir_patch.float(),gt_label
        else:
            _, file = os.path.split(self.image_filenames[index])


            if self.transform:

                input_im = self.transform(input_im)

                #r_h = H % 16
                #r_w = W % 16

            #if 0 != r_h or 0 != r_w:
            #    vis_input_im = self.resize(vis_input_im)


            return input_im.float(), vis_input_im.float(),nir_input_im.float(), gt_label, file

    def __len__(self):
        return len(self.image_filenames)


def transform():
    return Compose([
        # Resize((512,512)),
        # RandomHorizontalFlip(),
        # RandomVerticalFlip(),
        ToTensor(),
        # Normalize((0.5,), (0.5,))


    ])