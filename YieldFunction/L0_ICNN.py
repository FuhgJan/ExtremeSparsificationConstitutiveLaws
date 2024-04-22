
import numpy as np
import time
import torch
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D
import copy
#import Utility as util
from torch.autograd import grad
import math
#import randomConvex
import matplotlib.pyplot as plt

dev = torch.device('cpu')
if torch.cuda.is_available():
    print("CUDA is available, running on GPU")
    dev = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print("CUDA not available, running on CPU")

torch.manual_seed(2019)
E = 1000
nu = 0.3
mu = E / (2 * (1 + nu))
lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




from torch.autograd import Variable
BETA = 2 / 3
GAMMA = -0.1
ZETA = 1.1
EPSILON = 1e-6

class SymbolicLayerL0_Torch(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, weight_decay=1., droprate_init=0.5,
                 lamba=1., local_rep=False, **kwargs):
        super(SymbolicLayerL0_Torch, self).__init__()
        #super().__init__(funcs, initial_weight, variable, init_stddev)
        self.droprate_init = droprate_init if droprate_init != 0 else 0.5
        self.use_bias = bias
        self.lamba = lamba
        self.bias = None
        self.qz_log_alpha = None
        self.in_dim = in_features
        self.out_dim = out_features
        self.eps = None

        self.weight = torch.nn.Parameter(torch.Tensor(in_features, out_features))
        #self.register_parameter('weight', self.weight )

        #torch.nn.init.kaiming_normal(self.weight, mode='fan_out')
        self.qz_log_alpha = torch.nn.Parameter(torch.Tensor(in_features,out_features))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
            self.use_bias = True
        self.floatTensor = torch.FloatTensor   if not torch.cuda.is_available() else torch.cuda.FloatTensor
        self.qz_log_alpha.data.normal_(math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2)

        if self.use_bias:
            self.bias.data.fill_(0.1)



    def quantile_concrete(self, u):
        y = torch.nn.functional.sigmoid((torch.log(u) - torch.log(1.0-u) + self.qz_log_alpha) / BETA)
        return y * (ZETA - GAMMA) + GAMMA

    def sample_u(self, shape, reuse_u=False):
        if self.eps is None or not reuse_u:
            self.eps =  (EPSILON - (1.0 - EPSILON)) * torch.rand(shape) + 1.0 - EPSILON
        return self.eps




    def sample_z(self, batch_size, sample=True):
        if sample:
            eps = self.sample_u((batch_size, self.in_dim, self.out_dim))
            z = self.quantile_concrete(eps)
            return torch.clip_(z, 0., 1.)
        else:   # Mean of the hard concrete distribution
            pi = torch.nn.functional.sigmoid((self.qz_log_alpha))
            return torch.clip_(pi * (ZETA - GAMMA) + GAMMA, min=0.0, max=1.0)

    def get_z_mean(self):
        pi = torch.nn.functional.sigmoid(self.qz_log_alpha)
        return torch.clip_(pi * (ZETA - GAMMA) + GAMMA, min=0.0, max=1.0)

    def get_eps(self, size):
        eps = self.floatTensor(size).uniform_(epsilon, 1.-epsilon)
        eps = Variable(eps)

        return eps


    def sample_weights(self, reuse_u=False):
        z = self.quantile_concrete(self.sample_u((self.in_dim, self.out_dim), reuse_u=reuse_u))
        mask = torch.nn.functional.hardtanh(z, min_val=0., max_val=1.)
        y = self.weight * mask
        return y


    def get_weight(self):
        return self.weight * self.get_z_mean()

    def loss(self):
        return torch.sum(torch.nn.functional.sigmoid(self.qz_log_alpha - BETA * math.log(-GAMMA / ZETA)))


    def forward(self, x, sample=True, reuse_u=False):
        if sample:
            h = torch.matmul(x, self.sample_weights(reuse_u=reuse_u))
        else:
            w = self.get_weight()
            h = torch.matmul(x, w)

        if self.use_bias:
            h = h + self.bias

        return h






class l0_ICNN(torch.nn.Module):

    def __init__(self, inp, out, seedNumber, num_hidden_units=100, num_layers=2):


        torch.manual_seed(seedNumber)
        super(l0_ICNN, self).__init__()
        a=0.2
        b =2.5
        val_initialization = 0.1
        self.num_hidden_units = num_hidden_units
        self.fc1 = SymbolicLayerL0_Torch(inp, self.num_hidden_units, bias=False)
        self.beta1 = torch.nn.Parameter(torch.from_numpy((b - a) * np.random.uniform(size=1) + a).float().to(dev),
                                        requires_grad=True)

        torch.nn.init.trunc_normal_(self.fc1.weight.data, mean=0.0, std=1.0, a=0.0, b=val_initialization)

        self.fc2 = nn.ModuleList()

        self.fc2_betas = []
        for i in range(num_layers):
            self.fc2.append(SymbolicLayerL0_Torch(self.num_hidden_units+inp, num_hidden_units, bias=False))
            torch.nn.init.trunc_normal_(self.fc2[i].weight.data, mean=0.0, std=1.0, a=0.0, b=val_initialization)
            self.fc2_betas.append(
                torch.nn.Parameter(torch.from_numpy((b - a) * np.random.uniform(size=1) + a).float().to(dev),
                                   requires_grad=True))
            param_name = 'beta' + str(i + 2)
            self.register_parameter(param_name, self.fc2_betas[i])

        self.fc3 = SymbolicLayerL0_Torch(num_hidden_units+inp, out, bias=False)
        self.n_inp = inp
        self.loss_func = torch.nn.MSELoss()

        torch.nn.init.trunc_normal_(self.fc3.weight.data, mean=0.0, std=1.0, a=0.0, b=val_initialization)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3) #,weight_decay=1e-3)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.5)


    def getF(self, x, test=False):
        if test:
            y = self(x,sample=False,reuse_u=True)- self(torch.zeros(1, 2),sample=False,reuse_u=True) - torch.tensor(1.0)
        else:

            y = self(x,sample=False,reuse_u=True)- self(torch.zeros(1, 2),sample=False,reuse_u=True) - torch.tensor(1.0)


        return y

    def getLossRegL0(self):


        loss = self.fc1.loss() + self.fc3.loss()
        for fc in self.fc2:
            loss = loss + fc.loss()

        return loss

    def forward(self, x_in, sample=True,reuse_u=False):

        x = self.fc1(x_in, sample=sample, reuse_u=reuse_u)

        act = torch.nn.Softplus()



        x = act(x)
        iter = 0
        for fc in self.fc2:
            xcat = torch.cat((x, x_in), 1)
            x = fc(xcat, sample=sample, reuse_u=reuse_u)



            x = act(x)
            iter = iter + 1
        xcat = torch.cat((x, x_in), 1)

        x = self.fc3(xcat, sample=sample, reuse_u=reuse_u)
        return x


    def train_model(self, xi_tr, s_tr,
                    EPOCH):

        xi_tr = torch.from_numpy(xi_tr).float()
        xi_tr.requires_grad_(True)
        xi_tr = xi_tr.to(dev)

        s_tr = torch.from_numpy(s_tr).float()
        s_tr = s_tr.to(dev)

        epoch_list = []
        loss_tr_list = []
        param_G0_list = []

        par = count_parameters(self)
        param_G0_list.append(par)
        for name, W in self.named_parameters():
            print(name)


        for t in range(EPOCH):

            fout_tr = self.getF(xi_tr)

            l05_loss_w = torch.tensor(2e-4) * self.getLossRegL0()

            data_data = self.loss_func(fout_tr[:, 0], s_tr[:,0])
            loss_tr = data_data  + l05_loss_w

            self.optimizer.zero_grad()  # clear gradients for next train
            loss_tr.backward()  # backpropagation, compute gradients
            self.optimizer.step()  # apply gradients

            loss_tr_list.append(loss_tr.item())

            epoch_list.append(t)

            if t % 1 == 0:
                trainParam = []

                for name, W in self.named_parameters():
                    if 'weight' in name:
                        ww = W.data
                        trainParam.extend(ww.flatten().cpu().detach().numpy())
                trainParam = []

                self.beta1.data = self.beta1.data.clamp(min=0,max=1)

                for i, fc in enumerate(self.fc2):
                    ww = fc.weight.data
                    w = ww[:, :self.num_hidden_units].clamp(min=0)
                    fc.weight.data[:, :self.num_hidden_units] = w

                    self.fc2_betas[i].data = self.fc2_betas[i].data.clamp(min=0,max=1)


                ww = self.fc3.weight.data
                w = ww[:, :self.num_hidden_units].clamp(min=0)
                self.fc3.weight.data[:, :self.num_hidden_units] = w

                with torch.no_grad():
                    trainParam.extend(self.fc1.get_weight().flatten().cpu().detach().numpy())
                    trainParam.extend(self.fc3.get_weight().flatten().cpu().detach().numpy())
                    for i, fc in enumerate(self.fc2):
                        trainParam.extend(fc.get_weight().flatten().cpu().detach().numpy())

                trainParam_np = np.asarray(trainParam)
                idx_everything0 = np.where(np.abs(trainParam_np) > 1e-8)[0]

            if t > 0:
                param_G0_list.append(len(idx_everything0))

            if t % 1000 == 0:
                print(
                    'Iter: %d Loss: %.9e l05_loss_w: %.9e lData: %.9e'
                    % (
                        t + 1, loss_tr.item(), l05_loss_w.item(), data_data.item()))

                print('Iter: %d Number Parameters: %d' %
                      ( t + 1, len(idx_everything0)))


        return [epoch_list, loss_tr_list, param_G0_list]

    def saveModel(self, PATH):

        torch.save(self.state_dict(), PATH)

    def recoverModel(self, PATH):

        self.load_state_dict(torch.load(PATH))
