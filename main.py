from __future__ import print_function
from observations import multi_mnist
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from mlp_minst_model import SAIR
from dataset import *
import sys
from os.path import isfile
import os
import logging
is_train=True
task_catagory="training" if is_train else "inference"
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
train_data_address=sys.argv[2]
test_data_address=sys.argv[3]
z_pres_anneal_end_value=1e-5
z_pres_anneal_start_value=0.99
z_pres_anneal_start_step=1000
z_pres_anneal_end_step=100000

def anneal_z_pres_prob(prob, step):
    if z_pres_anneal_start_step < step < z_pres_anneal_end_step:
        slope = (z_pres_anneal_end_value - z_pres_anneal_start_value) / (z_pres_anneal_end_step - z_pres_anneal_start_step)
        prob = torch.tensor(z_pres_anneal_start_value + slope * (step - z_pres_anneal_start_step)).to(device)
    return prob

def train(epoch, model, train_loader, batch_size, optimizer):
    train_loss = 0
    num_samples = 60000
    for batch_idx, (data, _) in enumerate(train_loader):
        # print("batch_idx",batch_idx)
        data = data.view(1, -1, 50, 50)
        if(use_cuda):
            data = Variable(data).cuda()

        # forward + backward + optimize
        optimizer.zero_grad()
        kld_loss, nll_loss = model(data)
        loss = kld_loss + nll_loss
        if (batch_idx % 500 == 0):
            print("batch_idx=", batch_idx, "loss=", train_loss / (batch_size * (batch_idx + 1)))
        loss.backward()
        optimizer.step()

        nn.utils.clip_grad_norm_(model.parameters(), clip)

        # printing
        epoch_iters = num_samples // batch_size
        if batch_idx % epoch_iters == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t KLD Loss: {:.6f} \t NLL Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), num_samples,
                       100. * batch_idx / epoch_iters,
                       kld_loss.item() / batch_size,
                       nll_loss.item() / batch_size))

        train_loss += loss.item()

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / num_samples))

def train2(epoch, model, train_data,batch_size, model_optimizer,baseline_optimizer,global_step):
    """

    :param epoch:
    :param model:
    :param train_data: [10 60000 50 50]
    :param batch_size:
    :param optimizer:
    :return:
    """
    train_loss_baseline = 0
    train_loss_model=0
    num_samples = 60000
    for batch_idx in range(int(num_samples/batch_size)):
        global_step+=1
        data=train_data[:,batch_idx*batch_size:(batch_idx+1)*batch_size,:,:]#[10,batchsize,50,50]
        if (use_cuda):
            data = Variable(data).cuda()

            # forward + backward + optimize

        loss1, baseline_loss,_ = model(data)
        model.discovery.z_pres_prob = anneal_z_pres_prob(model.discovery.z_pres_prob, global_step)

        model_optimizer.zero_grad()
        loss1.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip)
        model_optimizer.step()
        """
        baseline_optimizer.zero_grad()
        baseline_loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip)
        baseline_optimizer.step()
        """
        #train_loss_baseline += baseline_loss.item()
        train_loss_model += loss1.item()
        #print(batch_idx)
        if (batch_idx % int((num_samples/batch_size)/3) == 0):
            print("batch_idx=", batch_idx, "train_loss_baseline=", train_loss_baseline / (batch_size * (batch_idx+1)),
            "train_loss_model=", train_loss_model / (batch_size * (batch_idx + 1)))




        # printing
        epoch_iters = num_samples // batch_size
        if batch_idx % epoch_iters == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t model loss: {:.6f} \t baseline loss: {:.6f}'.format(
                epoch, batch_idx * len(data), num_samples,
                       100. * batch_idx / epoch_iters,
                       loss1.item() / batch_size,
                       baseline_loss.item() / batch_size))



    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss_model / num_samples))
    return global_step


def fetch_data():
    inpath = 'data/multi_mnist/'
    (X_train, y_train), (X_test, y_test) = multi_mnist(inpath, max_digits=2, canvas_size=50, seed=42)
    X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)
    X_train /= 255.0
    X_test /= 255.0
    mnist_train = torch.from_numpy(X_train)
    mnist_test = torch.from_numpy(X_test)
    return mnist_train, y_train, mnist_test, y_test

def fetch_data2():
    #inpath = 'data/multi_mnist/'
    X_train,X_train_nums, X_test,X_test_nums = timestep_Multiminst(train_data_address,test_data_address)
    X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)
    X_train /= 255.0
    X_test /= 255.0
    mnist_train = torch.from_numpy(X_train)
    mnist_test = torch.from_numpy(X_test)
    mnist_test_nums=torch.from_numpy(X_train_nums)
    mnist_train_nums=torch.from_numpy(X_train_nums)
    return mnist_train, mnist_train_nums,mnist_test,mnist_test_nums

def test(epoch,model,mnist_test, mnist_test_nums, batch_size):
    """uses test data to evaluate
    likelihood of the model"""

    #model = SAIR()
    #_, _, mnist_test, mnist_test_nums = fetch_data2()
    train_loss_baseline = 0
    train_loss_model=0
    num_samples = mnist_test.size(1)
    true_label_num=0
    for batch_idx in range(int(num_samples / batch_size)):
        data = mnist_test[:, batch_idx * batch_size:(batch_idx + 1) * batch_size, :, :]  # [10,batchsize,50,50]
        nums_label=torch.reshape(mnist_test_nums[0,batch_idx * batch_size:(batch_idx + 1) * batch_size,:],(batch_size,3))#[B 3]
        nums_label=torch.matmul(nums_label,torch.tensor([[1],[1],[1]]).byte())# return numbers of digit in one batch [B 1]
        nums_label=nums_label.to(device)
        if (use_cuda):
            data = Variable(data).cuda()

            # forward + backward + optimize
        loss1, baseline_loss,z_pres = model(data)# z_pres[B T*max+1 1]
        train_loss_baseline += baseline_loss.item()
        train_loss_model += loss1.item()
        #print(z_pres.size())
        train_label=torch.sum(z_pres,dim=1)#[B 1]

        batch_label_same=torch.sum(torch.eq(nums_label,train_label.byte()),dim=0)#[1]
        #print("batch_label_same=",batch_label_same)
        true_label_num+=batch_label_same.item()
        if (batch_idx % 100 == 0):
            print("batch_idx=", batch_idx, "train_loss_baseline=", train_loss_baseline / (batch_size * (batch_idx+1)),
            "train_loss_model=", train_loss_model / (batch_size * (batch_idx + 1)))


        # printing
        epoch_iters = num_samples // batch_size
        if batch_idx % epoch_iters == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t KLD Loss: {:.6f} \t NLL Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), num_samples,
                       100. * batch_idx / epoch_iters,
                       loss1.item() / batch_size,
                       baseline_loss.item() / batch_size))


    print("Final_true_label_rate=", true_label_num / (batch_size * (int(num_samples / batch_size))), )
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss_model / num_samples))
def load_checkpoint(filename, model=None):
    print("loading model...")
    if (model):
        # model=torch.load(filename)
        model.load_state_dict(torch.load(filename))
    print("loading over")


if __name__ == "__main__":


    # hyperparameters
    n_epochs = 100
    clip = 1e-6
    model_lr = 1e-6
    baseline_lr=1e-4
    batch_size = 64
    seed = 128
    global_step=0

    # manual seed
    torch.manual_seed(seed)
    plt.ion()

    model = SAIR()
    """
    mnist_train, y_train, mnist_test, y_test = fetch_data()
    """
    mnist_train,mnist_train_nums, mnist_test,mnist_test_nums = fetch_data2()
    """
    train_dset = MultiMNIST_Dataset(mnist_train, y_train)
    test_dset = MultiMNIST_Dataset(mnist_test, y_test)
    
    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=True, num_workers=1)
    """

    # load model
    if isfile(sys.argv[1]):
        checkpoint = torch.load(sys.argv[1])
        model.load_state_dict(checkpoint['state_dict'])
        global_step=checkpoint['global_step']

    else:
        task_catagory="training"
        print("Not valid checkpoint, can not inference")
    if(task_catagory=="training"):
        for epoch in range(1, n_epochs + 1):
            model.train()
            # training + testing
            # learning_rate=(0.9**(epoch%10))*learning_rate
            model_optimizer = torch.optim.RMSprop(model.parameters(), lr=model_lr, momentum=0.9)
            baseline_optimizer = torch.optim.RMSprop(model.parameters(), lr=baseline_lr, momentum=0.9)
            global_step=train2(epoch, model, mnist_train, batch_size, model_optimizer,baseline_optimizer,global_step)
            torch.cuda.empty_cache()
            if(epoch%5==1):
                with torch.no_grad():
                    model.eval()
                    test(0, model, mnist_test, mnist_test_nums, batch_size)



            # saving model
            if epoch % 1 == 0:
                fn = '/content/drive/My Drive/SAIR-pytorch-version1/data/air_state_dict_' + str(epoch) + '.pth'
                #torch.save(model.state_dict(), fn)
                torch.save({'epoch': epoch,
                            'global_step': global_step,
                            'state_dict': model.state_dict()},
                           fn)
                print('Saved model to ' + fn)

    else:
        model.eval()
        with torch.no_grad():
              test(0, model, mnist_test,mnist_test_nums,batch_size)



