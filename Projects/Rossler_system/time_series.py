import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--init", nargs="+", type=float,  default=[-5.75, -1.6, 0.02])
value = parser.parse_args()

class ResNet1(nn.Module):
  def __init__(self):
    super(ResNet1, self).__init__()
    

    width = 30
    self.fc1 = nn.Linear(12, width)
    self.relu1 = nn.ReLU(inplace= True)
    self.fc2 = nn.Linear(width, width)
    
    self.fc3 = nn.Linear(width, width)
    self.relu2 = nn.ReLU(inplace= True)
    self.fc4 = nn.Linear(width, width)

    
    self.fc5 = nn.Linear(width, width)
    self.relu3 = nn.ReLU(inplace= True)
    self.fc6 = nn.Linear(width, width)
    
    self.fc7 = nn.Linear(width, width)
    self.relu4 = nn.ReLU(inplace= True)
    self.fc8 = nn.Linear(width, 3)

        
  def forward(self, x):
    x = self.fc1(x)
    x = self.relu1(x)
    x = self.fc2(x)
    x = self.fc3(x)
    x = self.relu2(x)
    x = self.fc4(x)
    x = self.fc5(x)
    x = self.relu3(x)
    x = self.fc6(x)
    x = self.fc7(x)
    x = self.relu4(x)
    x = self.fc8(x)
    return x

class Rossler_model:
    def __init__(self, delta_t):
        self.delta_t = delta_t #if discrete model your delta_t
                              #if continuous model chose one <=1e-2
        self.nb_steps = 10000//self.delta_t

        self.rosler_nn = ResNet1().to('cuda')
        self.rosler_nn.load_state_dict(torch.load('./model_final.pt'))
        
        self.initial_condition = np.array(value.init)
    
    def jacobian(self, v):
        x, z = v[0], v[2]
        a = 0.2
        c = 5.7
        res = np.array([[       0,      -1,       -1],
                       [        1,       a,        0],
                       [        z,       0,     x-c]])
        return res


    def full_traj(self, initial_condition): 
        # run your model to generate the time series with nb_steps
        # just the y cordinate is necessary. 

        self.rosler_nn.eval()


        output_list = []

        start_point = torch.Tensor(self.initial_condition)
        start_point = np.concatenate((start_point, self.jacobian(start_point).reshape(-1)))

        output = self.rosler_nn(torch.Tensor(start_point).cuda()).cpu().detach().numpy()
        for i in range(1000000):
            if i != 0:
                output = self.rosler_nn(torch.Tensor(t1).cuda()).cpu().detach().numpy()
            t1 = np.concatenate((output, self.jacobian(output).reshape(-1)))

            output_list.append(output)
        
        y = np.asarray(output_list)[:, -1]


        #if your delta_t is different to 1e-2 then interpolate y
        #in a discrete time array t_new = np.linspace(0,10000, 10000//1e-2)
        # y_new = interp1d(t_new, your_t, your_y)
        # I expect that y.shape = (1000000,)
        return y

    def save_traj(self,y):
        #save the trajectory in traj.npy file
        # y has to be a numpy array: y.shape = (1000000,)
          
        np.save('traj.npy',y)
        
    
if __name__ == '__main__':

    ROSSLER = Rossler_model(0.01)

    y = ROSSLER.full_traj(initial_condition=ROSSLER.initial_condition)

    ROSSLER.save_traj(y)

