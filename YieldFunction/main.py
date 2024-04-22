import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import L0_ICNN
import math

import torch
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from scipy import spatial

from mpl_toolkits.mplot3d import Axes3D
import sympy as sy
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def clockwiseangle_and_distance(point):
    origin = [0, 0]
    refvec = [0, 1]


    vector = [point[0]-origin[0], point[1]-origin[1]]
    # Length of vector: ||v||
    lenvector = math.hypot(vector[0], vector[1])
    if lenvector == 0:
        return -math.pi, 0
    normalized = [vector[0]/lenvector, vector[1]/lenvector]
    dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
    diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
    angle = math.atan2(diffprod, dotprod)

    if angle < 0:
        return 2*math.pi+angle, lenvector

    return angle, lenvector



def Cacazu_new_pi(pi_inp):
    pi_inp = np.concatenate((pi_inp,np.array((0.0)).reshape(1,)))
    A = np.array(((np.sqrt(2/3), - np.sqrt(1/6), - np.sqrt(1/6)), (0., np.sqrt(1/2), - np.sqrt(1/2)), (np.sqrt(1/3), np.sqrt(1/3), np.sqrt(1/3))))
    sig = np.dot(np.linalg.inv(A), pi_inp)
    sig_princ_np = np.sort(sig)

    sig = np.array(((sig_princ_np[0],0.0,0.0),(0.0,sig_princ_np[1],0.0),(0.0,0.0,sig_princ_np[2])))

    I1 = np.trace(sig)
    s = sig-(I1/3.)*np.eye(3)

    s_eig = np.linalg.eigvals(s).real
    s1 = s_eig[0]
    s2 = s_eig[1]
    s3 = s_eig[2]
    a = 2.0
    k = -0.5
    f = np.power(np.abs(s1) - k *s1,a)+np.power(np.abs(s2) - k *s2,a)+np.power(np.abs(s3) - k *s3,a)
    sig_vm = f-0.24

    return [sig_vm, 0.0]


def C_tens(lmbda,mu):
    dim = 3
    I = np.identity(dim)

    C = np.zeros((dim,dim,dim,dim))

    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for l in range(dim):
                    C[i,j,k,l] = lmbda*I[i,j]*I[k,l] + mu*(I[i,k]*I[j,l] + I[i,l]*I[j,k])
    return C


def getLinStress(eps_appl,lmbda,mu):
    dim =3
    C = C_tens(lmbda,mu)

    sig = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for l in range(dim):
                    sig[i,j] = sig[i,j] + C[i,j,k,l]*eps_appl[k,l]
    return sig



def batch_trace(X):
    trX = X.diagonal(offset=0, axis1=-1, axis2=-2).sum(-1)
    return trX

def batch_reshape(X):
    resh = np.expand_dims(X, axis=2)
    resh = resh * np.eye(X.shape[1])

    return resh



def getModelFromNet(pi_inp):
    pi_inp_torch = torch.as_tensor(pi_inp).float()
    pred = net.getF(pi_inp_torch.reshape(1,2),test=True).detach().cpu().numpy()
    f = [pred[0][0],0.0]
    return f


def getmodelFromSympy(pi_inp, f_sympy):
    pred = f_sympy(pi_inp[0],pi_inp[1])
    f = [pred,0.0]
    return f


def count_parameters(model):
    '''
    Args:
        model: Any model of type torch.nn.Module, probably works for other classes as well it just has to have model.parameters()
                predefined

    Returns: Number of paramters.

    '''
    # Reese: If you have a lazy layer it needs to have inferred the number
    # of parameters already, so you need to have a least run one forward pass
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




folder = './output'


ST = 'Cacazu_new_initial_data_30'
A = np.load('./../Data/'+ST+'.npz')

inp_pi = A['pi']


ST = ST + '_2'
center = np.zeros((1,3))

inp_pi_stacked = np.vstack((inp_pi,center))

out = np.zeros((inp_pi_stacked.shape[0],1))
out[-1] = -1.0


net = L0_ICNN.l0_ICNN(inp=2, out=1, seedNumber=0, num_hidden_units=50, num_layers=1)
print('Total param:   ',count_parameters(net))
outzero = net.getF(torch.zeros((1,2)))
trainFlag = True

modelpath = folder +ST+'_.pth'
if trainFlag:
    lists_afterTraining=net.train_model(inp_pi_stacked[:,0:2], out, 50000)

    net.saveModel(modelpath)
    filename = folder + ST+'_lists.npz'
    np.savez(filename, lists_afterTraining=lists_afterTraining)

    fig = plt.figure(figsize=(10, 8))
    # plt.gca().set_aspect('equal')
    ax1 = fig.add_subplot(1, 1, 1)
    # fig, ax1 = plt.subplots()

    color = 'k'
    ax1.set_xlabel('Epochs', fontsize=18)
    ax1.set_ylabel('Loss', color=color, fontsize=18)
    ax1.semilogy(lists_afterTraining[0][:], lists_afterTraining[1][:], color=color)
    ax1.tick_params(axis='y', labelsize=18, labelcolor=color)
    ax1.tick_params(axis='x', labelsize=16)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Number active param', color=color, fontsize=18)  # we already handled the x-label with ax1
    ax2.semilogy(lists_afterTraining[0][:], lists_afterTraining[2][:], color=color)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=18)
    ax2.set_ylim(1.)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(folder + ST+'_loss_Param.png')
    plt.close()


else:
    net.recoverModel(modelpath)


var_names = ['pi1','pi2']
vars = []
for var in var_names:
    if isinstance(var, str):
        vars.append(sy.Symbol(var))
    else:
        vars.append(var)



# Make data.
X = np.arange(-2, 2, 0.1)
Y = np.arange(-2, 2, 0.1)

X, Y = np.meshgrid(X, Y)
out = np.zeros_like(X)
out_pred =np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        out[i,j] = Cacazu_new_pi(np.array((X[i,j],Y[i,j])))[0]
        out_pred[i,j] = getModelFromNet(np.array((X[i,j],Y[i,j])))[0]


found_Samples = 0
pi_in_list = []


print('Find true function')
from scipy import optimize
from scipy.optimize import fsolve
while found_Samples<400:
    sample = np.random.uniform(low=-2., high=2., size=(2,))
    root = fsolve(Cacazu_new_pi, [sample[0],sample[1]] )
    val = Cacazu_new_pi(root)
    if np.abs(val[0]) < 1e-4:
        sig = np.sort(root)

        pi_in_list.append([root[0], root[1]])
        found_Samples = found_Samples + 1
        print(found_Samples)

pi_in_np = np.asarray(pi_in_list)
pts = pi_in_np.tolist()
s = sorted(pts, key=clockwiseangle_and_distance)
pi_in_np = np.asarray(s)
pi_in_np = np.vstack((pi_in_np,pi_in_np[0,:]))





found_Samples = 0
pi_in_list_NN = []

print('Find NN function')
from scipy import optimize
from scipy.optimize import fsolve
while found_Samples<400:
    sample = np.random.uniform(low=-2., high=2., size=(2,))
    root = fsolve(getModelFromNet, [sample[0],sample[1]] )
    val = getModelFromNet(root)
    if (np.abs(val[0]) < 1e-4):
        sig = np.sort(root)

        pi_in_list_NN.append([root[0], root[1]])
        found_Samples = found_Samples + 1
        print(found_Samples)




pi_in_NN_np = np.asarray(pi_in_list_NN)
pts = pi_in_NN_np.tolist()
s = sorted(pts, key=clockwiseangle_and_distance)
pi_in_NN_np = np.asarray(s)
pi_in_NN_np = np.vstack((pi_in_NN_np,pi_in_NN_np[0,:]))

fig = plt.figure(figsize=(10, 8))
# plt.gca().set_aspect('equal')
ax = fig.add_subplot(1, 1, 1)
plt.plot(pi_in_np[:, 0], pi_in_np[:, 1],lw=3.0, color='k', label='Ground truth',
                 linestyle='dashdot')
plt.plot(pi_in_NN_np[:, 0], pi_in_NN_np[:, 1],lw=2.0, label=r'$\hat{f}$')
plt.scatter(inp_pi[:,0],inp_pi[:,1], c= 'k')
ax.axhline(color='k', lw=1.2, ls='-')
ax.axvline(color='k', lw=1.2, ls='-')
ax.set_xlabel(r'$\pi_{1}$', fontsize=18)
ax.set_ylabel(r'$\pi_{2}$', fontsize=18)
plt.grid(color='0.95', linewidth=1.5)
plt.legend(fontsize=14, loc='upper left')
ST = ST + '.pdf'
plt.savefig(ST)
plt.close()



