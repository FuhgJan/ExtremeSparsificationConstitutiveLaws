
import numpy as np
import time
import torch
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D
import copy
#import Utility as util
from torch.autograd import grad
import math
import copy
#import randomConvex
import matplotlib.pyplot as plt

dev = torch.device('cpu')
if torch.cuda.is_available():
    print("CUDA is available, running on GPU")
    dev = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print("CUDA not available, running on CPU")


E = 1000
nu = 0.3
mu = E / (2 * (1 + nu))
lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))



def scSoftPlus(input,beta2):

    return (1/beta2)*torch.log(1.+beta2*torch.exp(input))


def scSoftPlus_Sec(input,beta2):
    return torch.log(1.+beta2*torch.exp(input))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)





limit_a, limit_b, epsilon = -.1, 1.1, 1e-6


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

        mask = torch.clip(z, min=0.0, max=1.0)
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


def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def is_psd(mat):
    eigs = torch.real(torch.linalg.eigvals(mat))
    return bool((eigs>=0).all())



class l0_ICNN(torch.nn.Module):
    def __init__(self, inp, out,  num_hidden_units=100, num_layers=2, enforceConv=True, seednumber=2019):

        super(l0_ICNN, self).__init__()

        torch.manual_seed(seednumber)
        self.enforceConv = enforceConv

        a=0.1
        b =0.3

        val_initialization = 0.1
        self.num_hidden_units = num_hidden_units
        self.fc1 = SymbolicLayerL0_Torch(inp, self.num_hidden_units, bias=False)
        self.beta1 = torch.nn.Parameter(torch.from_numpy((b - a) * np.random.uniform(size=1) + a).float().to(dev),
                                        requires_grad=True)
        a=0.01
        b =0.1
        if self.enforceConv:
            torch.nn.init.trunc_normal_(self.fc1.weight.data, mean=0.0, std=1.0, a=-val_initialization, b=val_initialization)
        else:
            torch.nn.init.trunc_normal_(self.fc1.weight.data, mean=0.0, std=1.0, a=-val_initialization, b=val_initialization)
        self.fc2 = nn.ModuleList()

        self.weights =[]
        self.qz = []

        self.weights += [self.fc1.weight]
        self.qz += [self.fc1.qz_log_alpha]
        self.fc2_betas = []
        for i in range(num_layers):
            self.fc2.append(SymbolicLayerL0_Torch(self.num_hidden_units+inp, num_hidden_units, bias=False))

            self.weights += [self.fc2[i].weight]
            self.qz += self.fc2[i].qz_log_alpha
            if self.enforceConv:
                torch.nn.init.trunc_normal_(self.fc2[i].weight.data, mean=0.0, std=1.0, a=0, b=val_initialization)
            else:
                torch.nn.init.trunc_normal_(self.fc2[i].weight.data, mean=0.0, std=1.0, a=-val_initialization, b=val_initialization)
            self.fc2_betas.append(
                torch.nn.Parameter(torch.from_numpy((b - a) * np.random.uniform(size=1) + a).float().to(dev),
                                   requires_grad=True))
            param_name = 'beta' + str(i + 2)
            self.register_parameter(param_name, self.fc2_betas[i])

        self.fc3 = SymbolicLayerL0_Torch(num_hidden_units+inp, out, bias=False)

        self.weights += [self.fc3.weight]
        self.qz += [self.fc3.qz_log_alpha]
        self.n_inp = inp
        self.loss_func = torch.nn.MSELoss()

        if self.enforceConv:
            torch.nn.init.trunc_normal_(self.fc3.weight.data, mean=0.0, std=1.0, a=0.0, b=val_initialization)
        else:
            torch.nn.init.trunc_normal_(self.fc3.weight.data, mean=0.0, std=1.0, a=-val_initialization, b=val_initialization)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-2) #,weight_decay=1e-5)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.5)



    def getpotential(self, x, test=False, reuse=False):
        undeformed_invariants = torch.tensor((3.0, 3.0)).reshape(1, 2).requires_grad_(True)
        J = torch.tensor(1.0).reshape(1, 1)  # x[:, 2:3]
        if test:
            psi0 = self(undeformed_invariants,sample=False,reuse_u=True)  # ,sample=False,reuse_u=True)

            dSdI = grad(psi0[:, 0].unsqueeze(1), undeformed_invariants,
                        torch.ones(undeformed_invariants.size()[0], 1, device=dev),
                        create_graph=True, retain_graph=True)[0]

            n = 2 * (dSdI[:, 0] + 2 * dSdI[:, 1])
            y = self(x ,sample=False,reuse_u=True) - psi0 - n * (J - 1)

        else:
            y_out = self(x ,sample=False,reuse_u=reuse)
            psi0 = self(undeformed_invariants ,sample=False,reuse_u=True)
            dSdI = grad(psi0[:, 0].unsqueeze(1), undeformed_invariants,
                        torch.ones(undeformed_invariants.size()[0], 1, device=dev),
                        create_graph=True, retain_graph=True)[0]

            n = 2 * (dSdI[:, 0] + 2 * dSdI[:, 1])
            y = y_out  - psi0 - n * (J - 1)

        return y.reshape((x.shape[0], 1))

    def getStressUniaxial(self, lambda1, test=False,reuse=False):

        I1 = lambda1 ** 2 + 2. / lambda1
        I2 = 2 * lambda1 + (1. / (lambda1 ** 2))

        IsoInvariants = torch.cat((I1, I2), dim=1)
        potential = self.getpotential(IsoInvariants, test, reuse=reuse)
        dPsidI = grad(potential[:, 0].unsqueeze(1), IsoInvariants, torch.ones(IsoInvariants.size()[0], 1, device=dev),
                      create_graph=True, retain_graph=True)[0]
        P1 = 2 * (dPsidI[:, 0:1] + (1. / lambda1) * dPsidI[:, 1:2]) * (lambda1 - (1. / (lambda1 ** 2)))
        return P1

    def getStressBiaxial(self, lambda1, test=False,reuse=False):

        I1 = 2 * (lambda1 ** 2) + (1. / (lambda1 ** 4))
        I2 = (lambda1 ** 4) + (2. / (lambda1 ** 2))

        IsoInvariants = torch.cat((I1, I2), dim=1)
        potential = self.getpotential(IsoInvariants, test, reuse=reuse)
        dPsidI = grad(potential[:, 0].unsqueeze(1), IsoInvariants, torch.ones(IsoInvariants.size()[0], 1, device=dev),
                      create_graph=True, retain_graph=True)[0]
        P1 = 2 * (dPsidI[:, 0:1] + (lambda1 ** 2) * dPsidI[:, 1:2]) * (lambda1 - (1. / (lambda1 ** 5)))
        return P1


    def getStressPS(self, lambda1, test=False,reuse=False):

        I1 =   (lambda1 ** 2) + torch.tensor(1.) + (1. / (lambda1 ** 2))
        I2 =  (lambda1 ** 2) + torch.tensor(1.) + (1. / (lambda1 ** 2))

        IsoInvariants = torch.cat((I1, I2), dim=1)
        potential = self.getpotential(IsoInvariants, test, reuse=reuse)
        dPsidI = grad(potential[:, 0].unsqueeze(1), IsoInvariants, torch.ones(IsoInvariants.size()[0], 1, device=dev),
                      create_graph=True, retain_graph=True)[0]
        P1 = 2 * (dPsidI[:, 0:1] + dPsidI[:, 1:2]) * (lambda1 - (1. / (lambda1 ** 3)))
        return P1

    def getLossRegL0(self):


        loss = self.fc1.loss() + self.fc3.loss()
        for fc in self.fc2:
            loss = loss + fc.loss()

        return loss

    def forward(self, x_in, sample=True,reuse_u=False):

        x = self.fc1(x_in, sample=sample, reuse_u=reuse_u)

        x = scSoftPlus_Sec(x, self.beta1)

        iter = 0
        for fc in self.fc2:
            xcat = torch.cat((x, x_in), 1)

            x = fc(xcat, sample=sample, reuse_u=reuse_u)
            x = scSoftPlus(x, self.fc2_betas[iter])

            iter = iter + 1

        xcat = torch.cat((x, x_in), 1)
        x = self.fc3(xcat, sample=sample, reuse_u=reuse_u)
        return x


    def get_weights(self):
        weights_fc2 = [self.fc2[i].get_weight().detach().cpu().numpy() for i in range(len(self.fc2))]

        avweights = [self.fc1.get_weight().detach().cpu().numpy()]+ weights_fc2 + [self.fc3.get_weight().detach().cpu().numpy()]
        return avweights



    def train_model(self, xi_tr, s_tr, xi_tr_BT,s_tr_BT,xi_tr_PS,s_tr_PS, EPOCH,regWeight,lr,optType='both',folder='./'):
        if optType=='both':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif optType=='':
            self.optimizer = torch.optim.Adam(self.weights, lr=lr) #,weight_decay=1e-5)

        xi_tr = torch.from_numpy(xi_tr).float()
        xi_tr.requires_grad_(True)
        xi_tr = xi_tr.to(dev)


        s_tr = torch.from_numpy(s_tr).float()
        s_tr = s_tr.to(dev)

        xi_tr_BT= torch.from_numpy(xi_tr_BT).float()
        xi_tr_BT.requires_grad_(True)
        xi_tr_BT = xi_tr_BT.to(dev)

        s_tr_BT = torch.from_numpy(s_tr_BT).float()
        s_tr_BT = s_tr_BT.to(dev)


        xi_tr_PS= torch.from_numpy(xi_tr_PS).float()
        xi_tr_PS.requires_grad_(True)
        xi_tr_PS = xi_tr_PS.to(dev)

        s_tr_PS = torch.from_numpy(s_tr_PS).float()
        s_tr_PS = s_tr_PS.to(dev)

        best_loss = np.inf

        epoch_list = []
        loss_tr_list = []
        param_G0_list = []

        par = count_parameters(self)
        param_G0_list.append(par)
        for name, W in self.named_parameters():
            print(name)

        lamda = torch.tensor(regWeight)
        for t in range(EPOCH):

            l05_loss_w = lamda * self.getLossRegL0()
            loss_BT = torch.tensor(0.0)
            r2_BT = torch.tensor(0.0)
            loss_UT = torch.tensor(0.0)
            r2_UT = torch.tensor(0.0)
            loss_PS = torch.tensor(0.0)
            r2_PS = torch.tensor(0.0)
            L = 1
            for l in range(L):
                bt_out = self.getStressBiaxial(xi_tr_BT,reuse=False)
                loss_BTl = self.loss_func(bt_out[:, 0], s_tr_BT)
                loss_BT = loss_BT+loss_BTl
                r2_BT = r2_BT+ r2_loss(bt_out[:, 0], s_tr_BT)

                fout_UT = self.getStressUniaxial(xi_tr,reuse=True)
                loss_UTl = self.loss_func(fout_UT[:,0], s_tr)
                loss_UT = loss_UT+loss_UTl
                r2_UT = r2_UT+ r2_loss(fout_UT[:, 0], s_tr)

                fout_PS = self.getStressPS(xi_tr_PS,reuse=True)
                loss_PSl = self.loss_func(fout_PS[:,0], s_tr_PS)
                loss_PS = loss_PS+loss_PSl
                r2_PS = r2_PS +  r2_loss(fout_PS[:, 0], s_tr_PS)

            loss_BT = loss_BT/L
            r2_BT = r2_BT/L
            loss_UT = loss_UT/L
            r2_UT = r2_UT/L
            loss_PS = loss_PS/L
            r2_PS = r2_PS/L
            loss_data =loss_BT+loss_UT#+loss_PS
            loss_tr = loss_data  + l05_loss_w

            self.optimizer.zero_grad()  # clear gradients for next train
            loss_tr.backward()  # backpropagation, compute gradients
            self.optimizer.step()  # apply gradients
            loss_tr_list.append([loss_data.item(), loss_BT.item(), loss_UT.item(), loss_PS.item(), r2_BT.item(), r2_UT.item(), r2_PS.item()])

            epoch_list.append(t)

            if t % 1 == 0:

                trainParam = []

                self.beta1.data = self.beta1.data.clamp(min=0.0001,max=2)


                for i, fc in enumerate(self.fc2):
                    if self.enforceConv:
                        ww = fc.weight.data
                        w = ww[:, :self.num_hidden_units].clamp(min=0)
                        fc.weight.data[:, :self.num_hidden_units] = w

                    self.fc2_betas[i].data = self.fc2_betas[i].data.clamp(min=0.0001,max=2)


                if self.enforceConv:
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
                param_G0_list.append(len(idx_everything0)+2)



            if t % 1000 == 0:

                print('Iter: %d R2BT: %.9e R2UT: %.9e R2PS: %.9e  '% (
                        t + 1, r2_BT.item(), r2_UT.item(),r2_PS.item()))

                print(
                    'Iter: %d Loss: %.9e l05_loss_w: %.9e LossData: %.9e  LossBT: %.9e  LossUT: %.9e   LossPS: %.9e'
                    % (
                        t + 1, loss_tr.item(), l05_loss_w.item(),loss_data.item(), loss_BT.item(), loss_UT.item(), loss_PS.item()))

                print('Iter: %d Number Parameters: %d' %
                      ( t + 1, len(idx_everything0)+2))



            if loss_tr < best_loss:
                best_loss = loss_tr
                state = copy.deepcopy(self.state_dict())



        self.load_state_dict(state)

        return [epoch_list, loss_tr_list, param_G0_list]

    def saveModel(self, PATH):

        torch.save(self.state_dict(), PATH)

    def recoverModel(self, PATH):

        self.load_state_dict(torch.load(PATH))
