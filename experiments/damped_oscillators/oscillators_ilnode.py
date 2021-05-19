import time
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import savemat


parser = argparse.ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--experiment_no', type=int, default=1)
parser.add_argument('--nhidden', type=int, default = 5)
parser.add_argument('--extra_dim', type=int, default = 3)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint
    


class ODEfunc(nn.Module):

    def __init__(self, dim, nhidden):
        super(ODEfunc, self).__init__()
        self.elu = nn.Tanh()
        self.fc1 = nn.Linear(dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, dim)
        self.nfe = 0

    def forward(self, t, z):
        self.nfe += 1
        out = self.fc1(z)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out
    
    
class ODEBlock(nn.Module):

    def __init__(self, odefunc, dim, integration_times):
        super(ODEBlock, self).__init__()
        self.il = nn.Linear(2,dim)
        self.odefunc = odefunc
        self.integration_times = integration_times
        self.outL = nn.Linear(dim, 2)

    def forward(self, x):
        x = self.il(x)
        out = odeint(self.odefunc, x, self.integration_times, rtol=args.tol, atol=args.tol)
        out = self.outL(out)
        return out

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value
       
        
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    

if __name__ == '__main__':
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    filename = 'ilnode('+str(args.extra_dim)+')./'+str(args.nhidden)+'./'
    try:
        os.makedirs('./'+filename)
    except FileExistsError:
        pass
    
    torch.random.manual_seed(2021) # Set random seed for repeatability package
    
    data_dim = 2 + args.extra_dim
    dim = data_dim
    #dim does not equal data_dim for ANODEs where they are augmented with extra zeros
 
    #download data
    z0 = torch.tensor(np.load('data/z0.npy')).float().to(device)
    z_in = torch.tensor(np.load('data/z.npy')).float().to(device)
    samp_ts = torch.tensor(np.load('data/samp_ts.npy')).float().to(device)
    
    cutoff = 30
    x0 = z0[:cutoff]
    v0 = z0[cutoff:]
    z0 = torch.cat((x0, v0), dim=1).to(device)
    
    z = torch.empty((int(len(samp_ts)), cutoff, 2)).to(device)
    
    for i in range(int(len(samp_ts))):
        xi = z_in[i][:cutoff]
        vi = z_in[i][cutoff:]
        z[i] = torch.cat((xi, vi), dim=1)
        
    nhidden = args.nhidden
    
    feature_layers = [ODEBlock(ODEfunc(dim, nhidden), dim, samp_ts)]
    model = nn.Sequential(*feature_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.MSELoss()
    
    itr_arr = np.empty(args.niters)
    loss_arr = np.empty(args.niters)
    nfe_arr = np.empty(args.niters)
    time_arr = np.empty(args.niters)

    # training
    start_time = time.time() 
    for itr in range(1, args.niters+1):
        feature_layers[0].nfe = 0
        iter_start_time = time.time()
        optimizer.zero_grad()
        #forward in time and solve ode
        pred_z = model(z0).to(device)
        # compute loss
        loss = loss_func(pred_z, z)
        loss.backward()
        optimizer.step()
        iter_end_time = time.time()
        # make arrays
        itr_arr[itr-1] = itr
        loss_arr[itr-1] = loss
        nfe_arr[itr-1] = feature_layers[0].nfe
        time_arr[itr-1] = iter_end_time-iter_start_time
        
        print('Iter: {}, running MSE: {:.4f}'.format(itr, loss))
            

    end_time = time.time()
    print('\n')
    print('Training complete after {} iterations.'.format(itr))
    loss = loss.detach().numpy()
    print('Train MSE = ' +str(loss))
    print('NFE = ' +str(feature_layers[0].nfe))
    print('Total time = '+str(end_time-start_time))
    print('No. parameters = '+str(count_parameters(model)))
    
    np.save(filename+'itr_arr.npy', itr_arr)
    np.save(filename+'nfe_arr.npy', nfe_arr)
    np.save(filename+'loss_arr.npy', loss_arr)
    np.save(filename+'time_arr.npy', time_arr)
    torch.save(model, filename+'model.pth')
    names = []
    params = []
    params_orig = []
    for name,param in model.named_parameters():
        names.append(name)
        params.append(param.detach().numpy())
        params_orig.append(param)
    for name,param in model.named_buffers():
        names.append(name)
        params.append(param.detach().numpy())
        
    if nhidden == dim: # For some reason, cannot save (n,n) matrix...
        w1 = params[0]
        b1 = params[1]
        w2 = params[2]
        b2 = params[3]
        w3 = params[4]
        b3 = params[5]
        w4 = params[6]
        b4 = params[7]
        w5 = params[8]
        b5 = params[9]
        nn1 = dict({'w1':w1,'b1':b1,'w2':w2,'b2':b2,'w3':w3,'b3':b3,'w4':w4,'b4':b4,'w5':w5,'b5':b5,'names':names,'mse':loss})
    else:
        nn1 = dict({'Wb':params,'names':names,'mse':loss})
                        
    savemat(filename+'model.mat',nn1)
    
       
        

        
    

        
      

        
    


