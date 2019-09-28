import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from module import *
from torch.nn.functional import grid_sample, affine_grid
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal

where_length=3
pres_length=1
what_length=50
hidden_size=256
max_number=3 # max latent variable number per fram
use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")
#hidden_state_size=256
MAX_STEP_DISCOVERY=3 # discovery max step
Batch_size=32

class SAIR(nn.Module):# whole SAIR model
    def __init__(self):
        super(SAIR, self).__init__()
        self.propgate=Propgate_time_step()
        self.discovery = Discovery_time_step()
        self.decode=decode_time_step()
    def forward(self, image):
        """
        :param image:[T B H W]
        :return:
        """
        T=image.size(0)
        B =image.size(1)
        kld_loss = 0
        nll_loss = 0

        # ini
        z_what=torch.ones(B, 1,what_length)
        z_where=torch.ones(B, 1,where_length)
        z_pres=torch.ones(B, 1,pres_length)
        hidden=torch.ones(B, 1,hidden_size)
        y=torch.zeros(B,image.size(2),image.size(3))#[B H W]
        for t in range(T):
            z_propagte_what, z_propagte_where, z_propagte_pres, h_temp_prop=self.propgate(image[t,:,:,:],z_what,z_where,z_pres,hidden)
            z_what, z_where, z_pres, loss1=self.discovery(image, z_propagte_what, z_propagte_where, z_propagte_pres)
            kld_loss+=loss1
            y=self.decode(y,z_what,z_where,z_pres)#[B H W]
            loss2=nn.functional.binary_cross_entropy(y, image[t,:,:,:], size_average=False)
            nll_loss+=loss2
        return  kld_loss, nll_loss








class decode_time_step(nn.Module):
    def __init__(self):
        super(decode_time_step, self).__init__()
        self.obj_decode = ObjectDecoder()
    def forward(self, img_prev,z_what,z_where,z_pres):# img_prev is same shape as img with all entries zeros.
        """

        :param img_prev:[B H W]
        :param z_where: [B  where_length]
        :param z_what: [B  what_length]
        :param z_pres: [B  pres_length]
        :return:
        """
        y_att = self.obj_decode(z_what)#[B 400]
        y = attentive_stn_decode(z_where, y_att)#[B 50 50]
        x = lay_obj_in_image(img_prev, y, z_pres)#[B 50 50]
        return x

class Discovery_time_step(nn.Module):
    def __init__(self,max_step,propgate_GlimpseEncoder):
        super( Discovery_time_step,self).__init__()
        self.max_step=max_step
        self.encoder_img=ObjectEncoder()
        self.glimpse_encoder = propgate_GlimpseEncoder()
        self.dis_rnncell=nn.LSTMCell(256)
        self.latent_predic_where_and_pres=Latent_Predictor_disc_where_and_pres(256)
        self.latent_predic_what=Latent_Predictor_disc_what()
    def _reparameterized_sample_where_and_pres(self,z_where_mu, z_where_sd,z_pres_proba,z_pres_prev):
        z_pres=Independent( Bernoulli(z_pres_proba * z_pres_prev), 1 ).sample()
        eps = torch.FloatTensor(z_where_sd.size()).normal_()
        eps = Variable(eps).to(device)
        return eps.mul(z_where_sd).add_(z_where_mu),z_pres  #[B where_length],[B pres_length]
    def latent_loss(self, z_mean, z_sig):
        mean_sq = z_mean * z_mean
        sig_sq = z_sig * z_sig
        return 0.5 * torch.mean(mean_sq + sig_sq - torch.log(sig_sq) - 1)#[1]
    def _reparameterized_sample_what(self, mean, std):
        #mean, std = self.latent_predic_what(input)
        if self.training:
            eps = torch.FloatTensor(std.size()).normal_()
            eps = Variable(eps).to(device)
            return eps.mul(std).add_(mean)
        else:
            return mean

    def forward(self, img, z_propagte_what,z_propagte_where,z_propagte_pres):
        """
        :param img: [B H W]
        :param z_propagte_where:[B number where_length]
        :param z_propagte_what:[B number what_length]
        :param z_propagte_pres:[B number pres_length]
        :return:
        """
        n = img.size(0)
        loss=0
        # initial
        h_dis=Variable(torch.zeros(n, 1, 256)).to(device)
        c_dis = Variable(torch.zeros(n, 1, 256)).to(device)
        z_pres = Variable(torch.ones(n, 1, 1)).to(device)
        z_where = Variable(torch.zeros(n, 1, 3)).to(device)
        z_what = Variable(torch.zeros(n, 1, 50)).to(device)

        e_t=self.encoder_img(img.view(img.size(0),50*50))
        for i in range(self.max_step):
            if(i==0):
                z_where_item=z_propagte_where[:,-1,:]
                z_what_item=z_propagte_what[:,-1,:]
                z_pres_item=z_propagte_pres[:,-1,:]
            else:
                z_where_item=z_where[:,i-1,:]
                z_what_item = z_what[:, i - 1, :]
                z_pres_item = z_pres[:, i - 1, :]
                h_dis,c_dis=dis_hidden_state(self.dis_rnncell,e_t,z_where_item,z_what_item,h_dis,c_dis)#[B 1 hidden_size]
            z_where_mu, z_where_sd, z_pres_proba=self.latent_predic_where_and_pres()

            loss+=self.latent_loss(z_where_mu, z_where_sd)
            z_where_item,z_pres_item=self._reparameterized_sample_where_and_pres(z_where_mu, z_where_sd, z_pres_proba,z_pres_item)
            x_att = attentive_stn_encode(z_where_item, img)  # Spatial trasform [B 400]
            encode = self.glimpse_encoder(x_att)  # [B 100]
            z_what_mean,z_what_std=self.latent_predic_what(encode)#[B 50]
            loss += self.latent_loss(z_what_mean,z_what_std)#[1]
            z_what_item=self._reparameterized_sample_what(z_what_mean,z_what_std)
            if(i==0):
                z_what=z_where_item.unsqueeze(1)
                z_where=z_where_item.unsqueeze(1)
                z_pres=z_pres_item.unsqueeze(1)
            else:
                z_what=torch.cat((z_what,z_what_item),dim=1)
                z_where = torch.cat((z_where, z_where_item), dim=1)
                z_pres = torch.cat((z_pres, z_pres_item), dim=1)
        return  z_what,z_where,z_pres,loss


class Propgate_time_step(nn.Module):#each time step propagate model
    def __init__(self):
        super(Propgate_time_step, self).__init__()
        self.relation_rnn=nn.LSTMCell(100+(where_length+what_length)*2+hidden_state_size, hidden_state_size)
        self.tem_rnn=nn.LSTMCell((50*50)+(3+50)*2+256, 256)
        self.pro_loca=nn.Linear(hidden_size,where_length)
        self.pro_loca1=nn.ReLU()
        self.glimpse_encoder=GlimpseEncoder()
        self.latent_predic_where=Latent_Predictor_prap(256+3)
        self.latent_predic_what = Latent_Predictor_prap(50+256+256)
        self.latent_predic_pres=Latent_Predictor_prap(50+3+256+256)


    def prop_loca(self,z_where_last,hidden_last):
        z_where_bias=self.pro_loca1(self.pro_loca(hidden_last))+z_where_last
        return z_where_bias                    # [B 3]

    def _reparameterized_sample_where(self, input):
        mean, std = self.latent_predic_where(input)
        if self.training:
            eps = torch.FloatTensor(std.size()).normal_()
            eps = Variable(eps).to(device)
            return eps.mul(std).add_(mean)
        else:
            return mean
    def _reparameterized_sample_pres(self, input,previous_pre):
        mean, std = self.latent_predic_pres(input)
        zpres=torch.matmul(Independent(Bernoulli(torch.matmul(mean,std))).sample(),previous_pre)
        return zpres#[B 1]

    def _reparameterized_sample_what(self, input):
        mean, std = self.latent_predic_what(input)
        if self.training:
            eps = torch.FloatTensor(std.size()).normal_()
            eps = Variable(eps).to(device)
            return eps.mul(std).add_(mean)
        else:
            return mean

    def forward(self, img,z_what_last_time,z_where_last_time,z_pres_last_time,hidden_last_time_temp):
      # img [B H W] input image
      # z_what_last_time [B numbers what_length]
      # z_where_last_time [B numbers where_length]
      # z_pres_last_time [B numbers pres_length]
      #
      # hidden_last_time_temp [B numbers hidden_size]
      n=img.size(0)
      numbers=z_what_last_time.size(1)

      #initilise
      h_rela = Variable(torch.zeros(n, 1, 256)).to(device)
      c_rela = Variable(torch.zeros(n, 1, 256)).to(device)
      h_temp = Variable(torch.zeros(n, 1, 256)).to(device)
      c_temp = Variable(torch.zeros(n, 1, 256)).to(device)
      z_pres = Variable(torch.ones(n, 1,1)).to(device)
      z_where = Variable(torch.zeros(n, 1,3)).to(device)
      z_what = Variable(torch.zeros(n, 1,50)).to(device)
      #

      for i in numbers:
          z_where_bias = self.prop_loca(z_where_last_time[:, i, :], hidden_last_time_temp[:, i, :])  # [B 3]
          x_att_bias = attentive_stn_encode(z_where_bias, img)  # Spatial trasform [B 400]
          encode_bias = self.glimpse_encoder(x_att_bias)  # [B 100]
          if (i != 0):
              h_rela_item, c_rela_item = relation_hidden_state(self.relation_rnn(), encode_bias, z_where_last_time[:, i, :],
                                                               z_where[:, i - 1, :]
                                                               , z_what_last_time[:, i, :], z_what[:, i - 1, :],
                                                               hidden_last_time_temp[:, i, :],
                                                               h_rela[:, i - 1, :], c_rela[:, i - 1, :])  # [B 1 256]
              torch.cat((h_rela, h_rela_item), dim=1)
              torch.cat((c_rela, c_rela_item), dim=1)
          elif (i == 0):
              h_rela, c_rela = relation_hidden_state(self.relation_rnn(), encode_bias, z_where_last_time[:, i, :],
                                                     z_where[:, i, :]
                                                     , z_what_last_time[:, i, :], z_what[:, i, :],
                                                     hidden_last_time_temp[:, i, :],
                                                     h_rela[:, i, :], c_rela[:, i, :])  # [B 1,256]

          z_where_cal=torch.cat((z_where_last_time[:,i,:],h_rela[:,i,:].squeeze(1)),2)#[B 3+256]
          z_where_item=self._reparameterized_sample(z_where_cal.squeeze(1))#[B 3]

          x_att=attentive_stn_encode(z_where_item, img) # Spatial trasform [B 400]
          encode=self.glimpse_encoder(x_att)  # [B 100]

          h_temp_item, c_temp_item = temp_hidden_state(self.relation_rnn(), encode,
                                                           z_where[:, i - 1, :],
                                                           hidden_last_time_temp[:, i, :],
                                                           h_rela[:, i , :], c_rela[:, i , :])  # [B 1 256]
          if(i!=0):
               torch.cat((h_temp, h_temp_item), dim=1)
               torch.cat((c_temp, c_temp_item), dim=1)
          else:
              h_temp=h_temp_item
              c_temp=c_temp_item

          z_what_cal=torch.cat((z_what_last_time[:,i,:],h_rela[:,i,:].squeeze(1),h_temp_item.squeeze(1)),1)#[B 50+256+256]
          z_what_item=self._reparameterized_sample_what(z_what_cal)
          z_pres_cal=torch.cat((z_what_item,z_where_item,h_rela[:,i,:].squeeze(1),h_temp_item.squeeze(1)),1)#[B 50+3+256+256]
          z_pres_item=self._reparameterized_sample_pres(z_pres_cal,z_pres_last_time[:,i,:].squeeze(1))#[B 1]
          if(i==0):
              z_pres=z_pres_item.unsqueeze(1)
              z_what=z_what_item.unsqueeze(1)
              z_where=z_where_item.unsqueeze(1)
          else:
              torch.cat((z_pres,z_pres_item.unsqueeze(1)),dim=1)
              torch.cat((z_where, z_where_item.unsqueeze(1)), dim=1)
              torch.cat((z_what, z_what_item.unsqueeze(1)), dim=1)
      return  z_what,z_where,z_pres,h_temp#[B number __length]











