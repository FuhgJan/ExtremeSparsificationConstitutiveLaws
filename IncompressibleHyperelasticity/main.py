import numpy as np
import torch
import csv
import L0_ICNN as l0_ICNN
import matplotlib.pyplot as plt
import glob
import sympy as sy

np.random.seed(0)
torch.manual_seed(0)
dev = torch.device('cpu')
if torch.cuda.is_available():
    print("CUDA is available, running on GPU")
    dev = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print("CUDA not available, running on CPU")



if __name__ == '__main__':

    BTdata = []
    with open('./../Data/treloar_ET_50.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            BTdata.append([float(row[0]), float(row[1])])
    BTdata_np = np.asarray(BTdata)


    F_BT = []
    P_BT = []
    for i in range(BTdata_np.shape[0]):
        F_BT.append(np.array(([BTdata_np[i,0],0,0],[0,BTdata_np[i,0],0],[0,0,(1/np.power(BTdata_np[i,0],2))])))
        P_BT.append(np.array(([BTdata_np[i,1],0,0],[0,BTdata_np[i,1],0],[0,0,0])))

    F_BT_np = np.asarray(F_BT)
    P_BT_np = np.asarray(P_BT)

    UTdata = []
    with open('./../Data/treloar_UT_50.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            UTdata.append([float(row[0]), float(row[1])])
    UTdata_np = np.asarray(UTdata)

    F_UT = []
    P_UT = []
    inpt_UT = []
    for i in range(UTdata_np.shape[0]):
        F_UT.append(np.array(([UTdata_np[i,0],0,0],[0,(1/np.sqrt(UTdata_np[i,0])),0],[0,0,(1/np.sqrt(UTdata_np[i,0]))])))
        P_UT.append(np.array(([UTdata_np[i,1],0,0],[0,0.0,0],[0,0,0])))
        F = F_UT[i]
        C = F.T @ F
        inpt_UT.append([np.trace(C), np.trace(np.linalg.det(C)*np.linalg.inv(C.T))])

    inpt_UT_np = np.asarray(inpt_UT)
    F_UT_np = np.asarray(F_UT)
    P_UT_np = np.asarray(P_UT)


    PSdata = []
    with open('./../Data/treloar_PS_50.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            PSdata.append([float(row[0]), float(row[1])])
    PSdata_np = np.asarray(PSdata)

    F_PS = []
    P_PS = []

    for i in range(PSdata_np.shape[0]):
        F_PS.append(np.array(([PSdata_np[i,0],0,0],[0,1.0,0],[0,0,(1/PSdata_np[i,0])])))
        P_PS.append(np.array(([0.0,PSdata_np[i,1],0],[0,0.0,0],[0,0,0])))

    F_PS_np = np.asarray(F_PS)
    P_PS_np = np.asarray(P_PS)




    net = l0_ICNN.l0_ICNN(inp=2, out=1, num_hidden_units=30, num_layers=1, enforceConv=True, seednumber=2018)


    lam = 5e-2

    folder = './output/'

    train_flag = True

    if train_flag:
        lists_afterTraining = net.train_model(UTdata_np[:, 0:1], P_UT_np[:, 0, 0], BTdata_np[:, 0:1], P_BT_np[:, 0, 0],
                                              PSdata_np[:, 0:1], P_PS_np[:, 0, 1], 50000, regWeight=lam, lr=1e-3,
                                              optType='both', folder=folder)

        net.saveModel(folder+'TrainedModel_Rubber.pt')
        filename = folder+'lists.npz'

    losses_np = np.asarray(lists_afterTraining[1])
    r2_bt =np.around(losses_np[-1,4], decimals=4)
    r2_ut =np.around(losses_np[-1,5], decimals=4)
    r2_ps =np.around(losses_np[-1,6], decimals=4)

    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(1, 1, 1)

    color = 'k'
    ax1.set_xlabel('Epochs', fontsize=18)
    ax1.set_ylabel('Loss', color=color, fontsize=18)
    ax1.semilogy(lists_afterTraining[0][:], losses_np[:,0], color=color)
    ax1.tick_params(axis='y', labelsize=18, labelcolor=color)
    ax1.tick_params(axis='x', labelsize=18)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('Number effective parameters', color=color, fontsize=18)  # we already handled the x-label with ax1
    ax2.semilogy(lists_afterTraining[0][:], lists_afterTraining[2], color=color)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=18)
    ax2.set_ylim(bottom=1.)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(folder+'loss_Param_Trel_50.png')
    plt.show()


    F_test_UT = []
    n=100
    lambUT = np.linspace(1.0,8.0,n) #np.max(F_UT_np[:,0,0]),n)
    for i in range(n):
        F_test_UT.append(np.array(([lambUT[i],0,0],[0,(1/np.sqrt(lambUT[i])),0],[0,0,(1/np.sqrt(lambUT[i]))])))

    F_test_UT_np = np.asarray(F_test_UT)
    lambUT_torch = torch.from_numpy(lambUT).to(dev)
    lambUT_torch.requires_grad_(True)
    stress_UT = net.getStressUniaxial(lambUT_torch.reshape((n,1)).float(),test=True)
    stress_UT_np = stress_UT.detach().cpu().numpy()


    F_test_BT = []
    n=100
    lambBT = np.linspace(1.0,5.8,n) #np.max(F_BT_np[:,0,0]),n)
    for i in range(n):
        F_test_BT.append(np.array(([lambBT[i],0,0],[0,lambBT[i],0],[0,0,(1/np.power(lambBT[i],2))])))

    F_test_BT_np = np.asarray(F_test_BT)
    lambBT_torch = torch.from_numpy(lambBT).to(dev)
    lambBT_torch.requires_grad_(True)
    stress_BT = net.getStressBiaxial(lambBT_torch.reshape((n,1)).float(),test=True)
    stress_BT_np = stress_BT.detach().cpu().numpy()


    F_test_ET = []
    n=100
    lambET = np.linspace(1.0,8.0,n)#np.max(F_PS_np[:,0,0]),n)
    for i in range(n):
        F_test_ET.append(np.array(([lambET[i],0,0],[0,1.,0],[0,0,(1/lambET[i])])))

    F_test_ET_np = np.asarray(F_test_ET)
    lambET_torch = torch.from_numpy(lambET).to(dev)
    lambET_torch.requires_grad_(True)
    stress_PS = net.getStressPS(lambET_torch.reshape((n,1)).float(),test=True)
    stress_PS_np = stress_PS.detach().cpu().numpy()


    #F_test_BT =
    fig = plt.figure(figsize=(10, 8))
    # plt.gca().set_aspect('equal')
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(F_UT_np[:,0,0],P_UT_np[:,0,0], label='UT test data', s=80, c='r', zorder=2)
    plt.scatter(F_BT_np[:,0,0],P_BT_np[:,0,0],c='b', label='ET test data', s=80, zorder=2)
    plt.scatter(F_PS_np[:,0,0],P_PS_np[:,0,1],c='g', label='PS test data', s=80, zorder=2)

    plt.plot(F_test_UT_np[:,0,0],stress_UT_np,c='r', label='ML UT fit  '+r'$R^{2}_{UT}=$'+str(r2_ut), lw=2.0, zorder=3)
    plt.plot(F_test_BT_np[:,0,0],stress_BT_np,c='b', lw=2.0, label='ML ET fit  '+r'$R^{2}_{ET}=$'+str(r2_bt), zorder=3)
    plt.plot(F_test_ET_np[:,0,0],stress_PS_np,c='g', lw=2.0, label='ML PS fit  '+r'$R^{2}_{PS}=$'+str(r2_ps), zorder=3)
    #plt.xlim([0.,8.0])
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)
    ax.set_xlabel(r'Stretch', fontsize=22)
    ax.set_ylabel(r'Nominal Stress [MPa]', fontsize=22)
    plt.grid(color='0.95', linewidth=1.5, zorder=1)
    plt.legend(fontsize=20, loc='upper left')
    fig.tight_layout()
    ST = folder+'MLTreloar_all_50.png'
    plt.savefig(ST)
    plt.show()

