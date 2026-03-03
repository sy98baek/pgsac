'''
Colorization + Color constancy joint learning
Colorization 먼저 400 epoch 까지 pretrain한 후에 color constancy subnet과 동시에 학습
C subnet의 encoder feature --> CC subnet의 decoder 전달 : C_CC_UNet_en_de
C subnet의 decoder feature --> CC subnet의 decoder 전달 : C_CC_UNet_de_de
C subnet의 encoder feature --> CC subnet의 encoder 전달 : C_CC_UNet_en_en
from model import ~~~ as C_CC_UNet 이부분만 바꿔주면 됨
'''
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch.nn.functional as F
import argparse
import cv2
import torch.backends.cudnn as cudnn
from dataloaders import *
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


import torch
from illuminant_gt import label, label2
from torch.autograd import Variable
import socket
import time
from loss import *

from torchvision.utils import save_image, make_grid
from torchvision.transforms import *
from pgsac import *


# Training settings
parser = argparse.ArgumentParser(description='UNET for colorization')
# parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--epoch_count', type=int, default=70, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=1000, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--lr', type=float, default=5e-5, help='initial learning rate for adam')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', default=True, help='use cuda?')
parser.add_argument('--gpu_mode', default=True, help='use cuda?')
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')

parser.add_argument('--checktest', default='checktest/Graph_CC_codebook_our/', type=str, help='Location to save checkpoint models')
parser.add_argument('--save_folder', default='weights/Graph_CC_codebook_our/', type=str, help='Location to save checkpoint models')
parser.add_argument('--data_dir', default='D:/dataset_1012/', type=str, help='Dataset dir')

parser.add_argument('--patch_size', type=int, default=128, help='patch size')
parser.add_argument('--batchSize', type=int, default=8, help='patch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='patch size')

parser.add_argument('--model_type', type=str, default='basic_UNet', help='patch size')
parser.add_argument('--resume', type=bool, default=False, help='patch size')
parser.add_argument('--start_iter', type=int, default=1, help='patch size')
parser.add_argument('--nEpochs_static', type=int, default=1, help='patch size')

parser.add_argument('--nEpochs', type=int, default=1000, help='patch size')
parser.add_argument('--snapshots', type=int, default=50, help='patch size')

opt = parser.parse_args()
device = torch.device('cuda:0')
# print(opt)


if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True

if not os.path.exists(opt.save_folder):
    os.makedirs(opt.save_folder)
if not os.path.exists(opt.checktest):
    os.makedirs(opt.checktest)

def log(text, LOGGER_FILE):
    with open(LOGGER_FILE, 'a') as f:
        f.write(text)
        f.close()

def normalize_image(tensor):
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    return tensor

def vector_rotate_img(img,delta_theta,delta_pi):
    ''' r,theta,pi로 이루어진 3채널을 입력값 delta_theta,delta_pi만큼 rotate시키는 함수 '''
    red, green, blue = img[:, 0, :, :].unsqueeze(1), img[:, 1, :, :].unsqueeze(1), img[:, 2, :, :].unsqueeze(1)
    r = torch.sqrt(red ** 2 + blue ** 2 + green ** 2)
    theta = torch.acos(blue / r)
    pi = torch.atan(green / red)
    output = torch.cat([r, theta, pi], dim=1)

    theta_1 = output[:, 1, :, :].unsqueeze(1) + torch.FloatTensor(delta_theta).repeat(output.size(0), 1, output.size(2), output.size(3))
    pi_1 = output[:, 2, :, :].unsqueeze(1) + torch.FloatTensor(delta_pi).repeat(output.size(0), 1, output.size(2), output.size(3))

    Red = r * torch.sin(theta_1) * torch.cos(pi_1)
    Green = r * torch.sin(theta_1) * torch.sin(pi_1)
    Blue = r * torch.cos(theta_1)

    return torch.cat([Red, Green, Blue], dim=1)

def vector_img(img):
    ''' RGB 3채널을 r,theta,pi 3채널로 변환하는 함수 '''
    red, green, blue = img[:, 0, :, :].unsqueeze(1), img[:, 1, :, :].unsqueeze(1), img[:, 2, :, :].unsqueeze(1)
    r = torch.sqrt(red ** 2 + blue ** 2 + green ** 2)
    theta = torch.acos(blue / r)
    pi = torch.atan(green / red)
    output = torch.cat([r, theta, pi], dim=1)
    output_1 = torch.where(torch.isnan(output.float()), torch.tensor(0.0, dtype=torch.float32).to(device), output)
    return output_1

def vector_reverse(r_theta_pi):
    ''' r,theta,pi로 이루어진 3채널을 RGB 3채널로 변환하는 함수 '''
    output = r_theta_pi

    r = output[:,0,:,:].unsqueeze(1)
    theta_1 = output[:, 1, :, :].unsqueeze(1) #+ torch.FloatTensor(delta_theta).repeat(output.size(0), 1, output.size(2), output.size(3))
    pi_1 = output[:, 2, :, :].unsqueeze(1) #+ torch.FloatTensor(delta_pi).repeat(output.size(0), 1, output.size(2), output.size(3))

    Red = r * torch.sin(theta_1) * torch.cos(pi_1)
    Green = r * torch.sin(theta_1) * torch.sin(pi_1)
    Blue = r * torch.cos(theta_1)

    return torch.cat([Red, Green, Blue], dim=1)

def train(epoch):

    epoch_loss = 0
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        # print(Variable(batch[0]))
        RGB_input,vis_input,nir_input,gt = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3])

        if cuda:
            RGB_input = RGB_input.to(device)
            wb = vis_input.to(device)
            nir = nir_input.to(device)
            gt = gt.to(device)

        optimizer.zero_grad()
        t0 = time.time()
        ''' new '''
        full_pred, local_illum, conf_map, full_illuminant, spec_fmap, adj, embedding_loss, z_q, perplexity, spectral_feature,embedding,min_encoding_indices = model(RGB_input, wb)

        full_loss = angular_loss(full_pred,gt)
        loss = full_loss + 0.25 * embedding_loss

        epoch_loss += loss.mean().item()
        loss.mean().backward(retain_graph=True)
        optimizer.step()
        t1 = time.time()
        print(f'===> Epoch[{epoch}]({iteration}/{len(training_data_loader)}):  total loss: {loss.mean()} cc loss:{full_loss}|| Timer: {t1-t0}')
    print(f'===> Epoch {epoch} Complete: Avg. Loss: {epoch_loss / len(training_data_loader)}')

def eval(epoch):
    angular_list=[]
    avg_cc = 0
    model.eval()
    for i,batch in enumerate(testing_data_loader):
        with torch.no_grad():

            RGB_input, vis, nir, gt, file = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3]), batch[4]#, batch[3]#, batch[3]

        if cuda:
            RGB_input = RGB_input.to(device)
            wb = vis.to(device)
            nir_input = nir.to(device)
            gt = gt.to(device)


        t0 = time.time()

        with torch.no_grad():
             pred, local_illum, conf_map, illum, spec_fmap, adj, embedding_loss, z_q, perplexity, spectral_feature, _, _ = model(RGB_input, wb)

        ''' calculate AE '''
        cc = angular_loss(pred, gt)
        avg_cc += cc
        angular_list.append(cc)
        print('{}번째 angular error = {}'.format(i, cc))
        t1 = time.time()
    angular_list1 = angular_list.copy()
    index_min = []
    index_max = []
    '''AE measure'''
    angular_list.sort()
    onefourth = len(testing_data_loader) // 4
    median = angular_list[len(testing_data_loader) // 2]
    maxerror, minerror = 0, 0
    # min 25% error
    for i in range(onefourth):
        minerror += angular_list[i]
        index_min.append(angular_list1.index(angular_list[i]))
    # max 25% error
    angular_list.sort(reverse=True)
    for i in range(onefourth):
        maxerror += angular_list[i]
        index_max.append(angular_list1.index(angular_list[i]))

    maxerror = maxerror / onefourth
    minerror = minerror / onefourth
    avg_cc = avg_cc / len(testing_data_loader)
    print('average:{} | median:{} | worst:{} | Best:{}'.format(avg_cc,median,maxerror,minerror))
    print("===> Processing Done")
    log('Epoch[{}] : average = {}| median:{} | worst:{} | Best:{}\n'.format(epoch,avg_cc,median,maxerror,minerror) , logfile)

def checkpoint(epoch):
    model_out_path = opt.save_folder + "/epoch_{}.pth".format(epoch)
    optimizer_path = opt.save_folder + "/optimizer_epoch_{}.pth".format(epoch)

    torch.save(optimizer.state_dict(),optimizer_path)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

def transform():
    return Compose([
        ToTensor(),
        # RandomRotation(degrees=45),  # -45도에서 45도 사이로 임의 회전
        RandomHorizontalFlip(p=0.5),  # 50% 확률로 수평 뒤집기
        RandomVerticalFlip(p=0.5)
    ])

class DualTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img1, img2):
        seed = torch.randint(0, 2**32, (1,)).item()  # 동일한 시드를 설정하여 동일한 변환 적용
        torch.manual_seed(seed)
        transformed_img1 = self.transform(img1)
        torch.manual_seed(seed)
        transformed_img2 = self.transform(img2)
        return transformed_img1, transformed_img2

def transform_test():
    return Compose([
        ToTensor(),
        Resize((256, 256))
    ])

if __name__ == '__main__':

    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)


    ''' Dataset '''
    print('===> Loading datasets')
    dual_transform = DualTransform(transform())

    training_set = DatasetFromFolder_our(opt.data_dir, gt_ilu=label, folder='train/', isTrain=True, transform=transform())
    training_data_loader = DataLoader(training_set, batch_size=opt.batchSize, shuffle=True, num_workers=0, drop_last=True)
    testing_set = DatasetFromFolder_our(opt.data_dir, gt_ilu=label2, transform=transform_test(), folder='test/', isTrain=False)
    testing_data_loader = DataLoader(testing_set, batch_size=1, shuffle=False, num_workers=0)

    print('===> Building model ', opt.model_type)

    ''' model '''
    model = Graph_CC_codebook(vis_ch=3,nir_ch=8)

    from loss import SimCLR_Loss
    # loss
    CC_loss = Angular_loss()
    L1_loss = nn.L1Loss()
    cs_loss = F.cosine_similarity
    simclr_loss = SimCLR_Loss(batch_size=opt.batchSize,temperature=0.5)

    if cuda:
        model = model.to(device)
        L1_loss = L1_loss.to(device)
        simclr_loss = simclr_loss.to(device)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4)

    def lambda_(epoch):
        return pow((1 - ((epoch) / opt.nEpochs)), 0.9)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_,verbose = True)

    logfile = opt.checktest + 'eval.txt'

    if opt.resume:  # resume from check point, load once-trained data
        # Load checkpoint
        weight_folder = sorted(os.listdir(opt.save_folder),key = lambda x : x.split('.')[0].split('_')[-1])
        opt.start_iter = int(weight_folder[-1].split('.')[0].split('_')[-1]) + 1
        checkpoint_name =opt.save_folder + '/' + weight_folder[-2]
        optimizer_name =opt.save_folder + '/' + weight_folder[-1]
        print('==> Resuming from checkpoint..')
        checkpoint_load = torch.load(checkpoint_name)
        optimizer_load = torch.load(optimizer_name)
        model.load_state_dict(checkpoint_load,strict = False)
        # optimizer.load_state_dict(optimizer_load)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"총 파라미터 수: {total_params}")
    log('dataset :\nmodel =\nlr =\n ', logfile)

    for epoch in range(opt.start_iter, opt.nEpochs + 1):
        print('러닝 레이트 : {}'.format([i['lr'] for i in optimizer.param_groups]))

        train(epoch)
        if epoch >100:
            scheduler.step()
        if (epoch + 1) % 10 == 0 :
            eval(epoch)
        if epoch % 100 == 0 :
            checkpoint(epoch)

