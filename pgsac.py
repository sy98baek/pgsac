from model.Unet import *
import torch
import torch.nn as nn
from torch.nn.functional import normalize
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange
from model.transformer import TransformerBlock
import math
from torchvision.transforms import Resize as RS

def tensor_to_image(tensor):
    if len(tensor.shape)==4 : #batch 포함
        b,c,h,w = tensor.size()
        tensor = tensor[0].squeeze()
        if c == 3:
            output = np.transpose(np.array(tensor), (1,2, 0))
            output = Image.fromarray(np.clip(output * 255.0, 0, 255.0).astype('uint8'))
        elif c == 1:
            output = np.transpose(np.array(tensor), (1, 0))
            output = Image.fromarray(np.clip(output * 255.0, 0, 255.0).astype('uint8'),mode='L')
    elif len(tensor.shape) == 3:
        c,h,w = tensor.size()
        tensor = tensor.squeeze()
        if c == 3:
            output = np.transpose(np.array(tensor), (1,2, 0))
            output = Image.fromarray(np.clip(output * 255.0, 0, 255.0).astype('uint8'))
        elif c == 1:
            output = np.transpose(np.array(tensor), (1, 0))
            output = Image.fromarray(np.clip(output * 255.0, 0, 255.0).astype('uint8'), mode='L')

    elif len(tensor.shape) == 2:
        h,w = tensor.size()
        output = np.transpose(np.array(tensor), (1, 0))
        output = Image.fromarray(np.clip(output * 255.0, 0, 255.0).astype('uint8'), mode='L')
    return output

# Some codes are borrowed from:
# https://github.com/tkipf/pygcn.git
# https://github.com/MishaLaskin/vqvae.git

class MSCNN(nn.Module):
    def __init__(self, ch):
        super(MSCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=ch, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.outlayer = nn.Linear(6048, 15)
        self.projection = nn.Conv2d(in_channels=96, out_channels=ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out1 = self.conv1(x)  ### channel 32
        # out1 = self.pool1(out1)
        out2 = self.conv2(out1)  ### channel 32
        # out2 = self.pool2(out2)
        out3 = self.conv3(out2)  ### channel 32
        # out3 = self.pool3(out3)

        out = torch.cat([
            out1, out2, out3
        ], dim=1)  ### channel 96

        # print(out1.shape, out2.shape, out3.shape)

        out_org = out
        out_finnal = self.projection(out)

        return out_finnal

class GraphConvolution_skip_codebook(nn.Module):
    """
    GCN with skip connection
    GCN(input) + input = output
    """

    def __init__(self, in_channels, out_channels, bias=True):
        super(GraphConvolution_skip_codebook, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, input, adj):
        """
        Forward computation assumes graphs have the same number of images
        in a given batch. Otherwise we should implement some masking tricks.

        Parameters
        ----------
        input : torch.Tensor
            Tensor shape should be [batch, n_imgs, ch, height, width]. 1,7,c,h,w
        adj : torch.Tensor
            Adjacency matrix (including self-loops) of the graph.
            Shape should be [batch, n_imgs, n_imgs]
        """
        ad_shape = adj.shape
        shape = input.shape

        adj = rearrange(adj, 'b (ad1 ad2) h_n w_n -> (b h_n w_n) ad1 ad2',ad1=8)
        #adj = rearrange(adj, 'b (ad1 ad2) h_n w_n -> (b h_n w_n) ad1 ad2', ad1=3)
        T = torch.sqrt(1 / adj.sum(dim=2))  # N,8
        # print('T ={}'.format(T.shape))
        D = torch.stack([torch.diag(x) for x in T])  # N,8,8
        # print('D ={}'.format(D))
        A_bar = (D @ adj @ D)  # 1,7,7
        # print('A_bar ={}'.format(A_bar))

        # input1 = input.reshape(-1, shape[2], shape[3], shape[4])  # Bx8,1,H,W  # Bx8xN,1,H,W
        # print('input_reshape ={}'.format(input.shape)) # 7,1024,16,16
        h = self.conv(input)
        assert not(torch.any(torch.isnan(h)))
        # print('h ={}'.format(h.shape)) # 7,1024,16,16
        # h = h.reshape(shape[0], shape[1], -1)  # B,7,CxHxW # 1,7,262144  # Bx8xN,1,H,W -> N, Bx8, HxW
        h = rearrange(h,'(b c n) o p1 p2 -> (b n) c (o p1 p2)',n=ad_shape[-1]*ad_shape[-2],c=8)
        #h = rearrange(h, '(b c n) o p1 p2 -> (b n) c (o p1 p2)', n=ad_shape[-1] * ad_shape[-2], c=3)
        # print('h_reshape ={}'.format(h.shape))
        output = (A_bar @ h)#.reshape(shape[0], shape[1], -1, shape[3], shape[4])
        output = rearrange(output,'(b n) c (o p1 p2) -> (b c n) o p1 p2',b=ad_shape[0],o=1,p1=shape[-2])
        # print('A_bar @ h = {}'.format( (A_bar@h).shape)) # 1,7,262144
        # print('output ={}'.format(output.shape)) # 1,7,1024,16,16

        return output + input

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_channels) + ' -> ' \
               + str(self.out_channels) + ')'

# Codes borrowed from https://github.com/MishaLaskin/vqvae/blob/master/models/residual.py
class ResidualLayer(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """

    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1,
                      stride=1, bias=False)
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x

class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim)]*n_res_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x

# Code borrowed from https://github.com/MishaLaskin/vqvae/blob/master/models/encoder.py
class Encoder(nn.Module):
    """
    This is the q_theta (z|x) network. Given a data sample x q_theta
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim): # 3 256 2 64
        super(Encoder, self).__init__()
        kernel = 4
        stride = 4
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel,
                      stride=stride, padding=0),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel,
                      stride=stride, padding=0),
            # nn.ReLU(),
            # nn.Conv2d(h_dim // 4, h_dim // 2, kernel_size=kernel,
            #           stride=stride, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel,
            #           stride=stride, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1,
            #           stride=stride-1, padding=1),
            ResidualStack(
                h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ReLU(),

        )

    def forward(self, x):
        return self.conv_stack(x)

# Code borrowed from https://github.com/MishaLaskin/vqvae/blob/master/models/quantizer.py
class VectorQuantizer1(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer1, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.fill_(1.0)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        device = 'cuda:0'
        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1).to(device)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(device) # 1920,64 zero tensor
        min_encodings.scatter_(1, min_encoding_indices, 1) # index에 맞는 곳에 1의 값을 넣는다 [.scatter_(dim,index,src)]
        # print('index adj:{min_encoding_indices[0]}')
        # get quantized latent vectors
        z_q1 = torch.matmul(min_encodings, self.embedding.weight.to(device)).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q1-z)**2) + self.beta * \
            torch.mean((z_q1 - z.detach().to(device)) ** 2)

        # preserve gradients
        z_q = z + (z_q1 - z.detach().to(device))#.detach().cpu()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))


        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        z_q1 = z_q1.permute(0, 3, 1, 2)
        return loss, z_q, perplexity, min_encodings, min_encoding_indices, self.embedding,z_q1

# Code borrowed from https://github.com/MishaLaskin/vqvae/blob/master/models/vqvae.py
class VQVAE_GCN(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, save_img_embedding_map=False):
        super(VQVAE_GCN, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer1(
            n_embeddings, embedding_dim, beta)


    def forward(self, x, verbose=False):

        z_e = self.encoder(x)

        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, perplexity, min_encodings,min_encoding_indices,embedding,z_q1 = self.vector_quantization(z_e)


        return embedding_loss, z_q, perplexity,embedding,min_encoding_indices

# Backbones
class Graph_attention_CC_skip_codebook(nn.Module):
    def __init__(self, vis_ch, nir_ch):
        super(Graph_attention_CC_skip_codebook, self).__init__()

        self.Local_illum_en = UNet_encoder(in_ch=vis_ch)
        self.local_illum_de = UNet_for_CC_decoder_dropout(out_ch=3)

        self.conf_en = UNet_encoder(in_ch=nir_ch)
        self.conf_de = UNet_for_CC_decoder_dropout(out_ch=1)
        self.mscnn = MSCNN(ch=vis_ch)

        self.gcn = []
        for _ in range(4):
            self.gcn.append(GraphConvolution_skip_codebook(in_channels=1, out_channels=1))
        self.gcn = nn.ModuleList(self.gcn)

        self.illum_conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.illum_conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1)

    def forward(self, RGB, vis, z_q):
        ''' spectral '''
        b, c, h, w = vis.shape
        spectral_original = vis
        patch_height,patch_width = 16,16
        patch_embed = rearrange(vis,'b c (h p1) (w p2) -> (b c h w) p1 p2', p1=patch_height, p2=patch_width).unsqueeze(1)
                                            # b 8 1 h w -> n b 8 h w
        adj = z_q # [n adj]
        ''' spatial'''
        spatial = RGB

        h = patch_embed
        n_patch = adj.shape[0]
        spec_fmap = [h.squeeze()]
        for graph_layer in self.gcn:
            h = torch.relu(graph_layer(h, adj))
            spec_fmap.append(h.squeeze())
        h = h + patch_embed
        h = h.squeeze()
        spectral_feature = rearrange(h, '(b c h w) p1 p2 -> b c (h p1) (w p2)',c=vis.shape[1],h=int(vis.shape[2]/16),w=int(vis.shape[3]/16))
        spectral_feature = spectral_feature * spectral_original

        ''' confidence network '''
        conf1, conf2, conf3, conf4, conf5 = self.conf_en(spectral_feature)
        conf = self.conf_de(conf5, conf4, conf3)
        conf_map_re = torch.reshape(conf, [conf.size(0), conf.size(2) * conf.size(3)])
        sum = torch.sum(conf_map_re, dim=1)
        sum = sum.reshape([sum.size(0), 1, 1, 1])
        conf = conf / (sum.repeat([1, 1, conf.size(2), conf.size(3)]) + 0.00001)
        conf_map = conf.repeat([1, 3, 1, 1])

        ''' local illuminant network '''
        # spatial
        spatial_feature = self.mscnn(spatial) * spatial
        L1, L2, L3, L4, L5 = self.Local_illum_en(spatial_feature)
        # local illuminant map
        local_illum = self.local_illum_de(L5, L4, L3)
        normal = (torch.norm(local_illum[:, :, :, :], p=2, dim=1, keepdim=True) + 1e-04)
        local_illum = local_illum[:, :, :, :] / normal

        ''' illuminant output '''
        local_il = local_illum * conf_map
        local_il = local_il.reshape(local_il.size(0), local_il.size(1), local_il.size(2) * local_il.size(3))
        weighted_sum = torch.sum(local_il, dim=2)
        pred = weighted_sum / (torch.norm(weighted_sum, dim=1, p=2, keepdim=True) + 0.000001)

        '''contrastive loss'''
        local_illum2 = self.illum_conv2(self.illum_conv1(local_illum))
        illum = normalize(torch.sum(torch.sum(local_illum2, 2), 2), dim=1)

        '''spectral feature'''
        spectral_feature = torch.mean(spectral_feature,dim=1)

        adj = rearrange(adj, 'b (ad1 ad2) h_n w_n -> (b h_n w_n) ad1 ad2', ad1=8)

        return pred, local_illum, conf_map, illum, spec_fmap, adj, spectral_feature

class Graph_CC_codebook(nn.Module):
    def __init__(self, vis_ch, nir_ch):
        super(Graph_CC_codebook, self).__init__()
        # encode image into continuous latent space
        self.VQ = VQVAE_GCN(h_dim = 256, res_h_dim = 64, n_res_layers = 2 ,n_embeddings = 20, embedding_dim = 8*8, beta=0.25) # n_embeddings = codebook의 token개수, embedding_dim = token의 크기
        self.CC = Graph_attention_CC_skip_codebook(vis_ch, nir_ch)


    def forward(self, RGB,vis):

        embedding_loss, z_q, perplexity,embedding,min_encoding_indices = self.VQ(RGB)
        pred, local_illum, conf_map, illum, spec_fmap, adj,spectral_feature = self.CC(RGB, vis, z_q)

        return pred, local_illum, conf_map, illum, spec_fmap, adj, embedding_loss, z_q, perplexity,spectral_feature,embedding,min_encoding_indices