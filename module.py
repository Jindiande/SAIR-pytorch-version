import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torch.nn.functional import grid_sample, affine_grid
from matplotlib.pyplot import *
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
'''
## Attention Window to latent code
## X_att -> z_what
'''
class ObjectEncoder(nn.Module):
    def __init__(self):
        super(ObjectEncoder, self).__init__()
        self.enc = nn.Sequential(nn.Linear(50*50, 400),
                                 nn.ReLU(),
                                 nn.Linear(400, 100))
        if(use_cuda):
           self=self.cuda()

    def forward(self, data):
        output = self.enc(data.view(data.size(0),50*50).cuda())

        return output#[B 100]


class GlimpseEncoder(nn.Module):
    def __init__(self):
        super(GlimpseEncoder, self).__init__()
        self.enc = nn.Sequential(nn.Linear(400, 200),
                                 nn.ReLU(),
                                 nn.Linear(200, 100))
        if(use_cuda):
           self=self.cuda()
    def forward(self, data):
        output=self.enc(data.to(device))
        return output #[B 100]


'''
## Reconstruct Attention Window from latent code
## z_what -> Y_att
'''


class ObjectDecoder(nn.Module):
    def __init__(self):
        super(ObjectDecoder, self).__init__()
        self.dec = nn.Sequential(nn.Linear(50, 200),
                                 nn.ReLU(),
                                 nn.Linear(200, 400),
                                 nn.Sigmoid())
        if(use_cuda):
           self=self.cuda()

    def forward(self, z_what):
        return self.dec(z_what.cuda())#[B number 400]


'''
## RNN hidden state to presence and location
## h -> z_pres, z_where
'''


class Latent_Predictor_prap(nn.Module):
    def __init__(self,input_size,output_size):
        super(Latent_Predictor_prap, self).__init__()
        self.pred = nn.Linear(input_size, (output_size*2))
        self._pres = nn.Sigmoid()
        if(use_cuda):
           self=self.cuda()


    def forward(self, h,output_size):
        z_param = self._pres(self.pred(h))
        mu = z_param[:, 0:output_size]
        sd = z_param[:, output_size:]
        return mu,sd
    
class Latent_Predictor_prap_pres(nn.Module):
    def __init__(self,input_size):
        super(Latent_Predictor_prap_pres, self).__init__()
        self.pred = nn.Linear(input_size, 1)
        self._pres = nn.Sigmoid()
        if(use_cuda):
           self=self.cuda()


    def forward(self, h):
        z_param = self._pres(self.pred(h))
        return z_param#[B 1]

class Latent_Predictor_disc_where_and_pres(nn.Module):
    def __init__(self,input_size):
        super(Latent_Predictor_disc_where_and_pres, self).__init__()
        self.pred = nn.Linear(input_size, (1+3+3))
        self._pres = nn.Sigmoid()
        self._where_mu = lambda x: x
        self._where_sd = nn.Softplus()
        if(use_cuda):
           self=self.cuda()

    def forward(self, h):
        z_param = self.pred(h)
        z_pres_proba = self._pres(z_param[:, 0:1])
        z_where_mu = self._where_mu(z_param[:, 1:4])
        z_where_sd = self._where_sd(z_param[:, 4:])
        return z_pres_proba, z_where_mu, z_where_sd


class Latent_Predictor_disc_what(nn.Module):
    def __init__(self):
        super(Latent_Predictor_disc_what, self).__init__()
        self.enc = nn.Sequential(nn.Linear(100, 400),
                                 nn.ReLU(),
                                 nn.Linear(400, 100))
        if(use_cuda):
           self=self.cuda()

    def forward(self, data):
        output = self.enc(data)#[B 100]
        z_where_mean=output[:,0:50]
        z_where_std=output[:,50:100]
        return z_where_mean,z_where_std#[B 50],[B 50]


'''
## Spatial Transformer to shift and scale the reconstructed attention window
## z_where, Y_att -> Y_i
'''


def expand_z_where(z_where):
    # Takes 3-dimensional vectors, and massages them into 2x3 matrices with elements like so:
    # [s,x,y] -> [[s,0,x],
    #             [0,s,y]]
    b = z_where.size(0)
    #n=z_where.size(1)
    expansion_indices = torch.LongTensor([1, 0, 2, 0, 1, 3])
    element=torch.zeros([1, 1]).expand(b, 1)
    if(use_cuda):
        expansion_indices = expansion_indices.cuda()
    #print("z_where.shape_expand",z_where.size())

    out = torch.cat((element.cuda(), z_where), 1)#[B L+1]
    return torch.index_select(out, 1, expansion_indices).view(b,2, 3)
"""
def expand_z_where_decode(z_where):
    # Takes 3-dimensional vectors, and massages them into 2x3 matrices with elements like so:
    # [s,x,y] -> [[s,0,x],
    #             [0,s,y]]
    
    #:param z_where:[b n where_length]
    #:return:
    
    b = z_where.size(0)
    n=z_where.size(1)
    expansion_indices = torch.LongTensor([1, 0, 2, 0, 1, 3])
    out = torch.cat((torch.zeros([1, 1]).expand(b, n,1), z_where), 2)#[B n L+1]
    return torch.index_select(out, 2, expansion_indices).view(b,n,2, 3)
"""

def attentive_stn_decode(z_where, obj):
    """

    :param z_where: [b where_length]
    :param obj: [B 400]
    :return:
    """
    b=z_where.size(0)
    n = z_where.size(1)

    theta = expand_z_where(z_where)#[b 2 3]
    grid = affine_grid(theta, torch.Size((b, 1, 50, 50)))#[b 50 50 2]
    out = grid_sample(obj.view(b, 1, 20, 20), grid)#[b 1 50 50]
    return out.view(b, 50, 50)


'''
## Spatial Transformer to obtain attention window
## z_where, X -> X_att
'''


def z_where_inv(z_where):
    # Take a batch of z_where vectors, and compute their "inverse".
    # That is, for each row compute:
    # [s,x,y] -> [1/s,-x/s,-y/s]
    # These are the parameters required to perform the inverse of the
    # spatial transform performed in the generative model.
    b = z_where.size(0)
    #n=z_where.size(1)
    out = torch.cat((torch.ones([1, 1]).type_as(z_where).expand(b, 1), -z_where[:, 1:]), 1)
    out = out / z_where[:,0:1]
    return out#[B where_length]


def attentive_stn_encode(z_where, image):
    """

    :param z_where: #[B where_length]
    :param image: #[B H W]
    :return:
    """
    b=z_where.size(0)
    theta_inv = expand_z_where(z_where_inv(z_where))
    grid = affine_grid(theta_inv.view(b,2,3), torch.Size((b, 1, 20, 20)))#[b 20 20 2]
    if(use_cuda):
        grid=grid.cuda()
    out = grid_sample(image.view(b, 1, 50, 50), grid)#[b 1 20 20]
    return out.view(b, -1)#[b 400]


'''
## Add objects to generate image
## Sum(Y_i) = Y
'''


def lay_obj_in_image(x_prev, y, z_pres):
    """

    :param x_prev: [B H W]
    :param y: [B 50 50]
    :param z_pres:[B pres_length]
    :return:
    """
    #print("x_prev",x_prev.size(),"y",y.size(),"z_pres",z_pres.size())
    x = x_prev + (y * z_pres.unsqueeze(2))
    x[x > 1] = 1.
    return x


'''
## RNN hidden state from image
## X -> h_i
'''

"""
def compute_hidden_state(rnn, data, z_where_prev, z_what_prev, z_pres_prev, h_prev, c_prev):
    n = data.size(0)  # [B 50 50]
    data = data.view(n, -1, 1).squeeze(2)  # [B 50*50]
    rnn_input = torch.cat((data, z_where_prev, z_what_prev, z_pres_prev), 1)
    h, c = rnn(rnn_input, (h_prev, c_prev))
    return h, c  # [B 256]
"""
# relation and temp rnn in class propagate
def relation_hidden_state(rnn, data,z_where_last_time,z_where_last_object,z_what_last_time,z_what_last_object,h_pre_temp,h_prev, c_prev):
    """

    :param rnn: LSTM()
    :param data: [B 100]
    :param z_where_last_time: [B 3]
    :param z_where_last_object: [B 3]
    :param z_what_last_time: [B 50]
    :param z_what_last_object: [B 50]
    :param h_pre_temp: [B 256]
    :param h_prev: [B 256]
    :param c_prev: [B 256]
    :return:
    """
    n = data.size(0)  # [B 10*10]
    #data = data.view(n, -1, 1).squeeze(1)  # [B 100]
    #print("data.size",data.size())
    rnn_input = torch.cat((data, z_where_last_time,z_where_last_object,z_what_last_time,z_what_last_object,h_pre_temp), 1)#[B 100+3*2+50*2+256]
    #print("RNN_INPUT",rnn_input.size())
    h, c = rnn(rnn_input, (h_prev, c_prev))
    return h.unsqueeze(1), c.unsqueeze(1)  # [B 1 256]
def temp_hidden_state(rnn, data,z_where_present_object,h_pre_temp,h_present_rela,c_prev):
    """
    
    :param rnn: 
    :param data: [B 100]
    :param z_where_present_object:[B 3] 
    :param h_pre_temp: [B 256]
    :param h_present_rela: [B 256]
    :param c_prev: [B 256]
    :return: 
    """
    n = data.size(0)  # [B 10*10]
    #data = data.view(n, -1, 1).squeeze(1)  # [B 100]
    rnn_input = torch.cat(
        (data, z_where_present_object, h_pre_temp, h_present_rela), 1)
    h, c = rnn(rnn_input, (h_pre_temp, c_prev))
    return h.unsqueeze(1), c.unsqueeze(1)  # [B 1 256]


#discovery rnn in class discovery
def dis_hidden_state(rnn,data,z_where_pres_object,z_what_pres_object,h_prev, c_prev):
    n = data.size(0)  # [B 10*10]
    #data = data.view(n, -1, 1).squeeze(1)  # [B 100]
    rnn_input = torch.cat(
        (data, z_where_pres_object, z_what_pres_object), 1)
    #print("z_where_pres_object.shape",z_where_pres_object.size(),"z_what_pres_object",z_what_pres_object.size())
    h, c = rnn(rnn_input, (h_prev.squeeze(1), c_prev.squeeze(1)))
    return h.unsqueeze(1), c.unsqueeze(1)  # [B 1 256]

def zeros(*args):
    x = torch.zeros(*args)
    return x.cuda() if use_cuda else x