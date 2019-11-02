import math
import torch
import torch.nn as nn
#from torch.autograd import Variable
from module import *
import torch.nn.functional as F
from torch.distributions.bernoulli import Bernoulli
#from torch.distributions.independent import Independent
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from torch.distributions.geometric import Geometric
where_length=3
pres_length=1
what_length=50
hidden_size=256
#max_number=1 # max latent variable number per fram
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
#hidden_state_size=256
MAX_STEP_DISCOVERY=2 # discovery max step


class SAIR(nn.Module):# whole SAIR model
    def __init__(self):
        super(SAIR, self).__init__()
        self.propgate=Propgate_time_step()
        self.discovery = Discovery_time_step(MAX_STEP_DISCOVERY)
        self.decode=decode_time_step()
        self.likelihood_sigma=0.3
        """
        if(use_cuda):
           self=self.cuda()
        """
        self = self.to(device)
    def forward(self, image):
        """
        :param image:[T B H W]
        :return:
        """
        image=Variable(image).to(device)
        T=image.size(0)
        B =image.size(1)
        loss = 0
        baseline_loss = 0

        # ini
        """
        if(use_cuda):
            z_what = torch.ones(B, 1, what_length).cuda()
            z_where = torch.ones(B, 1, where_length).cuda()
            z_pres = torch.ones(B, 1, pres_length).cuda()
            hidden = torch.ones(B, 1, hidden_size).cuda()
            y = torch.zeros(B, image.size(2), image.size(3)).cuda()  # [B H W]
        """
        z_what = torch.ones(B, 1, what_length).to(device)
        z_where = torch.ones(B, 1, where_length).to(device)
        z_pres = torch.ones(B, 1, pres_length).to(device)
        hidden = torch.ones(B, 1, hidden_size).to(device)
        y = torch.zeros(B, image.size(2), image.size(3)).to(device)  # [B H W]
        for t in range(T):
            #print("t=",t)
            z_propagte_what, z_propagte_where, z_propagte_pres, kl_z_what_prop,kl_z_where_prop,hidden=self.propgate(image[t,:,:,:],z_what,z_where,z_pres,hidden)
            z_what, z_where, z_pres, kl_z_pres,kl_z_where,kl_z_what,baseline,score_fn,hidden2=self.discovery(image[t,:,:,:], z_propagte_what, z_propagte_where, z_propagte_pres,self.propgate.glimpse_encoder)
            hidden=torch.cat((hidden,hidden2),dim=1)
            #kld_loss+=loss1
            #z_pres=Bernoulli(z_pres).sample()
            y=self.decode(y,z_what,z_where,z_pres)#[B H W]

            p_x_z = Normal(y.reshape(B,-1), self.likelihood_sigma)
            log_like = p_x_z.log_prob(image[t,:,:,:].reshape(B,-1)).sum(-1)  #[B 1]
            # --------------------
            # Compute variational bound and loss function
            # --------------------
            kl_z_what+=kl_z_what_prop
            kl_z_where+=kl_z_where_prop
            elbo = log_like  - kl_z_what - kl_z_where  # objective for loss function, but high variance [B 1]
            loss_item  = - torch.sum(elbo + (elbo - baseline).detach() * score_fn)  # var reduction surrogate objective objective (cf Mnih & Gregor NVIL)
            elbo=elbo.reshape(elbo.size(0),1)
            baseline_loss_item = F.mse_loss(elbo.detach(), baseline)
            loss+=loss_item
            baseline_loss+=baseline_loss_item

            #loss2=nn.BCELoss(size_average=False)
            #nll_loss+=loss2(torch.clamp(y,0,1), torch.clamp(image[t,:,:,:],0,1))
            #print("z_pres.size",z_pres.size())
        return  loss/T, baseline_loss/T,z_pres #[1],[1],[B,T*max_number+1,1]


class decode_time_step(nn.Module):
    def __init__(self):
        super(decode_time_step, self).__init__()
        self.obj_decode = ObjectDecoder()
        """
        if(use_cuda):
           self=self.cuda()
        """
        self=self.to(device)
    def forward(self, img_prev,z_what,z_where,z_pres):# img_prev is same shape as img with all entries zeros.
        """

        :param img_prev:[B H W]
        :param z_where: [B  number where_length]
        :param z_what: [B  number what_length]
        :param z_pres: [B  number pres_length]
        :return:
        """
        x=img_prev
        #print("decode_z_pres.shape",z_pres.size())
        for i in range(z_where.size(1)):
            y_att = self.obj_decode(z_what[:, i, :])  # [B 400]
            y = attentive_stn_decode(z_where[:,i,:], y_att)  # [B 50 50]
            x = lay_obj_in_image(x, y, z_pres[:,i,:])  # [B 50 50]
        return x

class Discovery_time_step(nn.Module):
    def __init__(self,max_step):
        super( Discovery_time_step,self).__init__()
        self.max_step=max_step
        self.encoder_img=ObjectEncoder()
        #self.glimpse_encoder = propgate_GlimpseEncoder()
        self.dis_rnncell=nn.LSTMCell(100+3+50,256)
        self.baseline_rnn=nn.LSTMCell(1+3+50+100,256)
        self.baseline_linear=nn.Linear(256,1)
        self.latent_predic_where_and_pres=Latent_Predictor_disc_where_and_pres(256)
        self.latent_predic_what=Latent_Predictor_disc_what()
        self.z_pres_prob=0.99
        """
        if(use_cuda):
           self=self.cuda()
        """
        self=self.to(device)

    def _reparameterized_sample_where_and_pres(self,z_where_mu, z_where_sd,z_pres_proba):
        z_pres = Bernoulli(z_pres_proba).sample()
        #z_pres=z_pres_proba
        eps =Normal(z_where_mu,z_where_sd).sample().to(device)

        return eps,z_pres  #[B where_length],[B pres_length]
    def latent_loss(self, z_mean, z_sig):
        mean_sq = z_mean * z_mean
        sig_sq = z_sig * z_sig
        return 0.5 * torch.mean(mean_sq + sig_sq - torch.log(sig_sq) - 1)#[1]
    def _reparameterized_sample_what(self, mean, std):
        #mean, std = self.latent_predic_what(input)
        if self.training:
            eps = Normal(mean,std).sample()
            return eps
        else:
            return mean

    def compute_geometric_from_bernoulli(self,obj_probs):  # [b MAX_STEP]
        """ compute a normalized truncated geometric distribution from a table of bernoulli probs
        args
            obj_probs -- tensor of shape (N, max_steps) of Bernoulli success probabilities.
        """
        cum_succ_probs = obj_probs.cumprod(1)
        fail_probs = 1 - obj_probs
        geom = torch.cat([fail_probs[:, :1], fail_probs[:, 1:] * cum_succ_probs[:, :-1], cum_succ_probs[:, -1:]],
                         dim=1)  # [B MAX_STEP+1]
        return geom / geom.sum(1, True)

    def compute_z_pres_kl(self,q_z_pres_geom, p_z_pres, writer=None):
        """ compute kl divergence between truncated geom prior and tabular geom posterior
        args
            p_z_pres -- torch.distributions.Geometric object
            q_z_pres_geom -- torch tensor of shape (N, max_steps + 1) of a normalized geometric pdf
        """
        # compute normalized truncated geometric
        p_z_pres_log_probs = p_z_pres.log_prob(
            torch.arange(q_z_pres_geom.shape[1], dtype=torch.float).to(device)).to(device)  # [max_steps + 1]
        p_z_pres_normed_log_probs = p_z_pres_log_probs - torch.exp(p_z_pres_log_probs).sum(dim=0).to(device)

        kl = q_z_pres_geom * (torch.log(q_z_pres_geom + 1e-8) - p_z_pres_normed_log_probs.expand_as(
            q_z_pres_geom))  # [B max_steps+1]
        return kl.to(device)

    def forward(self, img, z_propagte_what,z_propagte_where,z_propagte_pres,propgate_GlimpseEncoder):
        """
        :param img: [B H W]
        :param z_propagte_where:[B number where_length]
        :param z_propagte_what:[B number what_length]
        :param z_propagte_pres:[B number pres_length]
        :return:
        """
        #print("z_propagte_pres",z_propagte_pres.size(),"z_propagte_what.shape",z_propagte_what.size())
        n = img.size(0)
        loss=0
        # initial

        h_dis=Variable(torch.zeros(n, 1, 256)).to(device)
        c_dis = Variable(torch.zeros(n, 1, 256)).to(device)
        z_pres = Variable(torch.ones(n, 1, 1)).to(device)
        z_where = Variable(torch.zeros(n, 1, 3)).to(device)
        z_what = Variable(torch.zeros(n, 1, 50)).to(device)
        h_dis_item = Variable(torch.zeros(n, 1, 256)).to(device)

        kl_z_what = torch.zeros(n, device=device)
        kl_z_where = torch.zeros(n, device=device)
        obj_probs = torch.ones(n, self.max_step, device=device)

        h_baseline = torch.zeros(n, 256, device=device)
        c_baseline = torch.zeros_like(h_baseline)
        baseline=torch.zeros(n,1).to(device)
        """
        h_dis = zeros(n, 1, 256)
        c_dis = zeros(n, 1, 256)
        z_pres = zeros(n, 1, 1)
        z_where = zeros(n, 1, 3)
        z_what = zeros(n, 1, 50)
        h_dis_item= zeros(n, 1, 256)
        """
        if(use_cuda):
            e_t=self.encoder_img(img.view(img.size(0),50*50).to(device))#[B 100]
        else:
            e_t = self.encoder_img(img.view(img.size(0), 50 * 50))  # [B 100]
        for i in range(self.max_step):
            if(i==0):
                z_where_item=z_propagte_where[:,-1,:]
                z_what_item=z_propagte_what[:,-1,:]
               # z_pres_item=z_propagte_pres[:,-1,:]
            else:
                z_where_item=z_where[:,i-1,:]
                z_what_item = z_what[:, i - 1, :]
                #z_pres_item = z_pres[:, i - 1, :]
            h_dis_item,c_dis=dis_hidden_state(self.dis_rnncell,e_t,z_where_item,z_what_item,h_dis_item,c_dis)#[B 1 hidden_size]
            z_pres_proba,z_where_mu, z_where_sd=self.latent_predic_where_and_pres(h_dis_item.squeeze(1))
            #print("z_pres_proba_discovery", z_pres_proba.size())
            loss+=self.latent_loss(z_where_mu, z_where_sd)
            #print("z_pres_item_discovery",z_pres_item.size())
            z_where_item,z_pres_item=self._reparameterized_sample_where_and_pres(z_where_mu, z_where_sd, z_pres_proba)
            x_att = attentive_stn_encode(z_where_item, img)  # Spatial trasform [B 400]
            encode = propgate_GlimpseEncoder(x_att)  # [B 100]
            z_what_mean,z_what_std=self.latent_predic_what(encode)#[B 50]
            loss += self.latent_loss(z_what_mean,z_what_std)#[1]
            z_what_item=self._reparameterized_sample_what(z_what_mean,z_what_std)
            if(i==0):
                z_what=z_what_item.unsqueeze(1)
                z_where=z_where_item.unsqueeze(1)
                z_pres=z_pres_item.unsqueeze(1)
                h_dis=h_dis_item
            else:
                z_what=torch.cat((z_what,z_what_item.unsqueeze(1)),dim=1)
                z_where = torch.cat((z_where, z_where_item.unsqueeze(1)), dim=1)
                z_pres = torch.cat((z_pres, z_pres_item.unsqueeze(1)), dim=1)
                h_dis=torch.cat(( h_dis,h_dis_item), dim=1)

            baseline_input = torch.cat([e_t.view(n,-1).detach(), z_pres_item.detach(), z_what_item.detach(), z_where_item.detach()], dim=-1)# [B,1+3+50+100]
            h_baseline, c_baseline = self.baseline_rnn(baseline_input, (h_baseline, c_baseline))#[B 256]
            #print("test self.baseline_linear(h_baseline).squeeze()=",self.baseline_linear(h_baseline).size())
            baseline += self.baseline_linear(h_baseline)  # note: masking by z_pres give poorer results [B 1]


            kl_z_what += kl_divergence(Normal(z_what_mean,z_what_std), Normal(torch.zeros(50).to(device),torch.ones(50).to(device))).sum(1) * z_pres_item.squeeze()#[B 1]
            kl_z_where += kl_divergence(Normal(z_where_mu,z_where_sd), Normal(torch.tensor([0.3, 0., 0.]).to(device),torch.tensor([0.1, 1., 1.]).to(device))).sum(1) * z_pres_item.squeeze()#[B 1]

            #pred_counts[:, i] = z_pres_item.flatten()  # [b MAX_STEP] binary
            obj_probs[:, i] = z_pres_proba[:,0] # [b MAX_STEP] z_pres_proba[b 1]


        q_z_pres = self.compute_geometric_from_bernoulli(obj_probs).to(device)
        #print("torch.arange(n)", torch.arange(n).type())
        score_fn = q_z_pres[
            torch.arange(n).long(), z_pres.long().squeeze(2).sum(1)].log()  # log prob of num objects under the geometric
        #print("z_pres.long()",z_pres.long().type())
        kl_z_pres = self.compute_z_pres_kl(q_z_pres, Geometric(torch.tensor([1-self.z_pres_prob]).to(device))).sum(
            1)  # [B 1]


        z_what = torch.cat((z_propagte_what, z_what), dim=1)
        z_where = torch.cat((z_propagte_where, z_where), dim=1)
        z_pres = torch.cat((z_propagte_pres, z_pres), dim=1)
        return  z_what,z_where,z_pres,kl_z_pres,kl_z_where,kl_z_what,baseline,score_fn,h_dis


class Propgate_time_step(nn.Module):#each time step propagate model
    def __init__(self):
        super(Propgate_time_step, self).__init__()
        self.relation_rnn=nn.LSTMCell(100+(where_length+what_length)*2+hidden_size, hidden_size)
        self.tem_rnn=nn.LSTMCell(256*2+100+3, 256)
        self.pro_loca=nn.Linear(hidden_size,where_length)
        self.pro_loca1=nn.ReLU()
        self.glimpse_encoder=GlimpseEncoder()
        self.latent_predic_where=Latent_Predictor_prap(256+3,where_length)
        self.latent_predic_what = Latent_Predictor_prap(50+256+256,what_length)
        self.latent_predic_pres=Latent_Predictor_prap_pres(50+3+256+256)
        """
        if(use_cuda):
           self=self.cuda()
        """
        self=self.to(device)


    def prop_loca(self,z_where_last,hidden_last):
        z_where_bias=self.pro_loca1(self.pro_loca(hidden_last))+z_where_last
        return z_where_bias                    # [B 3]

    def _reparameterized_sample_where(self, input):
        mean, std = self.latent_predic_where(input,where_length)#[B 3]
        eps = Normal(mean, std).sample()
        return eps,mean,std
    def _reparameterized_sample_pres(self, input,previous_pre):
        """

        :param input:
        :param previous_pre: [B 1]
        :return:
        """
        para= self.latent_predic_pres(input)#[B 1]
        #print("pres_prev_prop_shape", previous_pre.size())
        zpres=Bernoulli(para*previous_pre).sample()
        #zpres=para*previous_pre
        zpres=zpres.to(device)
        return zpres#[B 1]

    def _reparameterized_sample_what(self, input):
        mean, std = self.latent_predic_what(input,what_length)#[B 50]
        eps = Normal(mean, std).sample()
        return eps,mean,std

    def forward(self, img,z_what_last_time,z_where_last_time,z_pres_last_time,hidden_last_time_temp):
      # img [B H W] input image
      # z_what_last_time [B numbers what_length]
      # z_where_last_time [B numbers where_length]
      # z_pres_last_time [B numbers pres_length]
      #
      # hidden_last_time_temp [B numbers hidden_size]
      n=img.size(0)
      numbers=z_what_last_time.size(1)
      #print("hidden_last_time_temp.size",hidden_last_time_temp.size())

      #initilise
      """
      h_rela = zeros(n, 1, 256)
      c_rela = zeros(n, 1, 256)
      h_temp = zeros(n, 1, 256)
      c_temp = zeros(n, 1, 256)
      z_pres = zeros(n, 1, 1)
      z_where = zeros(n, 1,3)
      z_what = zeros(n, 1,50)
      #
      """
      h_rela = Variable(torch.zeros(n, 1, 256)).to(device)
      c_rela = Variable(torch.zeros(n, 1, 256)).to(device)
      h_temp = Variable(torch.zeros(n, 1, 256)).to(device)
      c_temp = Variable(torch.zeros(n, 1, 256)).to(device)
      z_pres = Variable(torch.zeros(n, 1, 1)).to(device)
      z_where = Variable(torch.zeros(n, 1, 3)).to(device)
      z_what = Variable(torch.zeros(n, 1, 50)).to(device)

      kl_z_what = torch.zeros(n, device=device)
      kl_z_where = torch.zeros(n, device=device)
      #print("numbers",numbers)
      for i in range(numbers):
          #print("i=",i)
          z_where_bias = self.prop_loca(z_where_last_time[:, i, :], hidden_last_time_temp[:, i, :])  # [B 3]
          x_att_bias = attentive_stn_encode(z_where_bias, img)  # Spatial trasform [B 400]
          encode_bias = self.glimpse_encoder(x_att_bias)  # [B 100]
          if (i != 0):
              h_rela_item, c_rela_item = relation_hidden_state(self.relation_rnn, encode_bias, z_where_last_time[:, i, :],
                                                               z_where[:, i - 1, :]
                                                               , z_what_last_time[:, i, :], z_what[:, i - 1, :],
                                                               hidden_last_time_temp[:, i, :],
                                                               h_rela[:, i - 1, :], c_rela[:, i - 1, :])  # [B 1 256]
              h_rela=torch.cat((h_rela, h_rela_item), dim=1)
              c_rela=torch.cat((c_rela, c_rela_item), dim=1)
          elif (i == 0):
              #print("test2")
              h_rela, c_rela = relation_hidden_state(self.relation_rnn, encode_bias, z_where_last_time[:, i, :],
                                                     z_where[:, i, :]
                                                     , z_what_last_time[:, i, :], z_what[:, i, :],
                                                     hidden_last_time_temp[:, i, :],
                                                     h_rela[:, i, :], c_rela[:, i, :])  # [B 1 256]
          #print("h_rela",h_rela.size())
          z_where_cal=torch.cat((z_where_last_time[:,i,:],h_rela[:,i,:]),1)#[B 3+256]
          z_where_item,z_where_mean,z_where_std=self._reparameterized_sample_where(z_where_cal)#[B 3]
          #print("z_where_item",z_where_item.size())
          x_att=attentive_stn_encode(z_where_item, img) # Spatial trasform [B 400]
          encode=self.glimpse_encoder(x_att)  # [B 100]

          h_temp_item, c_temp_item = temp_hidden_state(self.tem_rnn, encode,
                                                           z_where[:, i - 1, :],
                                                           hidden_last_time_temp[:, i, :],
                                                           h_rela[:, i , :], c_rela[:, i , :])  # [B 1 256]
          if(i!=0):
              h_temp=torch.cat((h_temp, h_temp_item), dim=1)
              c_temp=torch.cat((c_temp, c_temp_item), dim=1)
          else:
              h_temp=h_temp_item
              c_temp=c_temp_item

          z_what_cal=torch.cat((z_what_last_time[:,i,:],h_rela[:,i,:],h_temp_item.squeeze(1)),1)#[B 50+256+256]
          z_what_item,z_what_mean,z_what_std=self._reparameterized_sample_what(z_what_cal)#[B 50]
          #print("z_what_item.shape",z_what_item.size())
          z_pres_cal=torch.cat((z_what_item,z_where_item,h_rela[:,i,:],h_temp_item.squeeze(1)),1)#[B 50+3+256+256]
          #print("z_pres_cal.shape",z_pres_cal.size())
          z_pres_item=self._reparameterized_sample_pres(z_pres_cal,z_pres_last_time[:,i,:])#[B 1]
          if(i==0):
              z_pres=z_pres_item.unsqueeze(1)
              z_what=z_what_item.unsqueeze(1)
              z_where=z_where_item.unsqueeze(1)
          else:
              z_pres=torch.cat((z_pres,z_pres_item.unsqueeze(1)),dim=1)
              z_where=torch.cat((z_where, z_where_item.unsqueeze(1)), dim=1)
              z_what=torch.cat((z_what, z_what_item.unsqueeze(1)), dim=1)
          kl_z_what += kl_divergence(Normal(z_what_mean, z_what_std),
                                     Normal(torch.zeros(50).to(device), torch.ones(50).to(device))).sum(
              1) * z_pres_item.squeeze()  # [B 1]
          kl_z_where += kl_divergence(Normal(z_where_mean, z_where_std), Normal(torch.tensor([0.3, 0., 0.]).to(device),
                                                                             torch.tensor([0.1, 1., 1.]).to(
                                                                                 device))).sum(
              1) * z_pres_item.squeeze()  # [B 1]

      #print("z_pres_prop_shape",z_pres.size())
      #print("h_temp")
      return  z_what,z_where,z_pres,kl_z_what,kl_z_where,h_temp#[B number __length]