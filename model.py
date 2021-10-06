import torch 
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import numpy as np
from math import *
import torch.nn as nn


class DeepRitzNet(torch.nn.Module):
    def __init__(self, args):
        super(DeepRitzNet, self).__init__()
        self.linear0 = torch.nn.Linear(args.input_dim, args.hidden_dim)
        self.linear1 = torch.nn.Linear(args.hidden_dim,args.hidden_dim)
        self.linear2 = torch.nn.Linear(args.hidden_dim,args.hidden_dim)
        self.linear3 = torch.nn.Linear(args.hidden_dim,args.hidden_dim)
        self.linear4 = torch.nn.Linear(args.hidden_dim,args.hidden_dim)
        self.linear5 = torch.nn.Linear(args.hidden_dim,args.hidden_dim)
        self.linear6 = torch.nn.Linear(args.hidden_dim,args.hidden_dim)
        
        self.linear7 = torch.nn.Linear(args.hidden_dim,1)
      
    def forward(self, x):
        # skip connection every two layers
        # use ReLU as activation

        # # first use a linear transformation to map x to match the hidden dim
        # y = F.relu(self.linear0(x))
        # # apply residual blocks
        # y = y + F.relu(self.linear2(F.relu(self.linear1(y))))
        # y = y + F.relu(self.linear4(F.relu(self.linear3(y))))
        # y = y + F.relu(self.linear6(F.relu(self.linear5(y))))
        # # final linear layer to map output to a scalar
        # output = self.linear7(y)

        # first use a linear transformation to map x to match the hidden dim
        y = torch.tanh(self.linear0(x))
        # apply residual blocks
        y = y + torch.tanh(self.linear2(torch.tanh(self.linear1(y))))
        y = y + torch.tanh(self.linear4(torch.tanh(self.linear3(y))))
        y = y + torch.tanh(self.linear6(torch.tanh(self.linear5(y))))
        # final linear layer to map output to a scalar
        output = self.linear7(y)

        return output
    
    def cube(self, x):
        # return torch.max(x**3, 0)[0]
        return F.relu(x)**3
    

# class RitzNet(torch.nn.Module):
#     def __init__(self, args):
#         super(RitzNet, self).__init__()
#         self.linearIn = nn.Linear(args.input_dim, args.hidden_dim)
#         self.linear = nn.ModuleList()
#         for _ in range(args.num_layers):
#             self.linear.append(nn.Linear(args.hidden_dim, args.hidden_dim))

#         self.linearOut = nn.Linear(args.hidden_dim, 1)

#     def forward(self, x):
#         x = torch.tanh(self.linearIn(x)) # Match dimension
#         for layer in self.linear:
#             x = torch.tanh(layer(x))
        
#         return self.linearOut(x)


class RitzNet(torch.nn.Module):
    def __init__(self, args):
        super(RitzNet, self).__init__()
        # network layers
        self.linearIn = nn.Linear(args.input_dim, args.hidden_list_u[0])
        self.linear = nn.ModuleList()
        for i in range(len(args.hidden_list_u)-1):
            self.linear.append(nn.Linear(args.hidden_list_u[i], args.hidden_list_u[i+1]))

        self.linearOut = nn.Linear(args.hidden_list_u[-1], 1)

        # activation function
        if args.act_u == 'tanh':
            self.act = torch.nn.Tanh()
        elif args.act_u == 'softplus':
            # if tau=0, it means we want to use the principled approach: 
            # tau=sqrt(m), m: total number of params
            if args.tau_sp < 1e-8:
                args.tau_sp = np.sqrt(self.get_num_params())
            self.act = torch.nn.Softplus(beta=args.tau_sp)

    def forward(self, x):
        x = self.act(self.linearIn(x)) # Match dimension
        for layer in self.linear:
            x = self.act(layer(x))
        
        return self.linearOut(x)
    
    def get_num_params(self):
        return sum([p.numel() for p in self.parameters()])



class U_Net(torch.nn.Module):
    def __init__(self, args):
        super(U_Net, self).__init__()
        # network layers
        self.linearIn = nn.Linear(args.input_dim, args.hidden_list_u[0])
        self.linear = nn.ModuleList()
        for i in range(len(args.hidden_list_u)-1):
            self.linear.append(nn.Linear(args.hidden_list_u[i], args.hidden_list_u[i+1]))

        self.linearOut = nn.Linear(args.hidden_list_u[-1], 1)

        # activation function
        if args.act_u == 'tanh':
            self.act = torch.nn.Tanh()
        elif args.act_u == 'softplus':
            # if tau=0, it means we want to use the principled approach: 
            # tau=sqrt(m), m: total number of params
            if args.tau_sp < 1e-8:
                args.tau_sp = np.sqrt(self.get_num_params())
            self.act = torch.nn.Softplus(beta=args.tau_sp)

    def forward(self, t, x):
        x = torch.cat((t,x), dim=-1)
        x = self.act(self.linearIn(x)) # Match dimension
        for layer in self.linear:
            x = self.act(layer(x))
        
        return self.linearOut(x)
    
    def get_num_params(self):
        return sum([p.numel() for p in self.parameters()])


class U_Net_vec(torch.nn.Module):
    def __init__(self, args):
        super(U_Net_vec, self).__init__()
        # network layers
        self.linearIn = nn.Linear(args.input_dim, args.hidden_list_u[0])
        self.linear = nn.ModuleList()
        for i in range(len(args.hidden_list_u)-1):
            self.linear.append(nn.Linear(args.hidden_list_u[i], args.hidden_list_u[i+1]))

        self.linearOut = nn.Linear(args.hidden_list_u[-1], args.d)

        # activation function
        if args.act_u == 'tanh':
            self.act = torch.nn.Tanh()
        elif args.act_u == 'softplus':
            # if tau=0, it means we want to use the principled approach: 
            # tau=sqrt(m), m: total number of params
            if args.tau_sp < 1e-8:
                args.tau_sp = np.sqrt(self.get_num_params())
            self.act = torch.nn.Softplus(beta=args.tau_sp)

    def forward(self, t, x):
        x = torch.cat((t,x), dim=-1)
        x = self.act(self.linearIn(x)) # Match dimension
        for layer in self.linear:
            x = self.act(layer(x))
        
        return self.linearOut(x)
    
    def get_num_params(self):
        return sum([p.numel() for p in self.parameters()])


class U_Net_functional(torch.nn.Module):
    def __init__(self, args):
        super(U_Net_functional, self).__init__()
        self.params_size = []
        self.cml_size    = []
        self.args        = args
        num_params       = self.compute_param_sizes()
        if args.act_u == 'tanh':
            self.act = torch.nn.Tanh()
        elif args.act_u == 'softplus':
            if args.tau_sp < 1e-8:
                args.tau_sp = np.sqrt(num_params)
            self.act = torch.nn.Softplus(beta=args.tau_sp)

        print ("Number of parameters in U-Net: ", num_params)

    def forward(self, x, params):
        # input layer
        # x     : (N, *, in_dim)
        # weight: (out_dim, in_dim)
        # bias  : (out_dim)
        x = self.act(nn.functional.linear(input=x, \
            weight=params[:self.cml_size[0]].view(self.args.hidden_list_u[0], self.args.input_dim), \
            bias=params[self.cml_size[0]:self.cml_size[1]].view(-1)))
        
        # hidden layers
        for i in range(len(self.args.hidden_list_u)-1):
            x = self.act(nn.functional.linear(input=x, \
                weight=params[self.cml_size[2*i+1]:self.cml_size[2*i+2]].view(self.args.hidden_list_u[i+1], self.args.hidden_list_u[i]), \
                bias=params[self.cml_size[2*i+2]:self.cml_size[2*i+3]].view(-1)))
        
        # output layer
        return nn.functional.linear(input=x, \
            weight=params[self.cml_size[-3]:self.cml_size[-2]].view(1, self.args.hidden_list_u[-1]), \
            bias=params[self.cml_size[-2]:self.cml_size[-1]].view(-1))
    
    def compute_param_sizes(self):
        # input layer, A and b
        self.params_size.append(self.args.input_dim*self.args.hidden_list_u[0])
        self.params_size.append(self.args.hidden_list_u[0])
        # hidden blocks
        for i in range(len(self.args.hidden_list_u)-1):
            self.params_size.append(self.args.hidden_list_u[i]*self.args.hidden_list_u[i+1])
            self.params_size.append(self.args.hidden_list_u[i+1])
        # output layer
        self.params_size.append(self.args.hidden_list_u[-1]*1)
        self.params_size.append(1)

        self.cml_size = np.cumsum(self.params_size)

        return sum(self.params_size)
        
    def get_num_params(self):
        return sum(self.params_size)




class U_Net_vec_functional(torch.nn.Module):
    def __init__(self, args):
        super(U_Net_vec_functional, self).__init__()
        self.params_size = []
        self.cml_size    = []
        self.args        = args
        num_params       = self.compute_param_sizes()
        if args.act_u == 'tanh':
            self.act = torch.nn.Tanh()
        elif args.act_u == 'softplus':
            if args.tau_sp < 1e-8:
                args.tau_sp = np.sqrt(num_params)
            self.act = torch.nn.Softplus(beta=args.tau_sp)

        print ("Number of parameters in U-Net: ", num_params)

    def forward(self, x, params):
        # input layer
        # x     : (N, *, in_dim)
        # weight: (out_dim, in_dim)
        # bias  : (out_dim)
        x = self.act(nn.functional.linear(input=x, \
            weight=params[:self.cml_size[0]].view(self.args.hidden_list_u[0], self.args.input_dim), \
            bias=params[self.cml_size[0]:self.cml_size[1]].view(-1)))
        
        # hidden layers
        for i in range(len(self.args.hidden_list_u)-1):
            x = self.act(nn.functional.linear(input=x, \
                weight=params[self.cml_size[2*i+1]:self.cml_size[2*i+2]].view(self.args.hidden_list_u[i+1], self.args.hidden_list_u[i]), \
                bias=params[self.cml_size[2*i+2]:self.cml_size[2*i+3]].view(-1)))
        
        # output layer
        return nn.functional.linear(input=x, \
            weight=params[self.cml_size[-3]:self.cml_size[-2]].view(1, self.args.hidden_list_u[-1]), \
            bias=params[self.cml_size[-2]:self.cml_size[-1]].view(-1))
    
    def compute_param_sizes(self):
        # input layer, A and b
        self.params_size.append(self.args.input_dim*self.args.hidden_list_u[0])
        self.params_size.append(self.args.hidden_list_u[0])
        # hidden blocks
        for i in range(len(self.args.hidden_list_u)-1):
            self.params_size.append(self.args.hidden_list_u[i]*self.args.hidden_list_u[i+1])
            self.params_size.append(self.args.hidden_list_u[i+1])
        # output layer - output a vector that matches the ambient dimension
        self.params_size.append(self.args.hidden_list_u[-1]*self.args.input_dim)
        self.params_size.append(self.args.input_dim)

        self.cml_size = np.cumsum(self.params_size)

        return sum(self.params_size)
        
    def get_num_params(self):
        return sum(self.params_size)




class P_Net(torch.nn.Module):
    def __init__(self, args, u_net_size):
        super(P_Net, self).__init__()
        # network layers
        # input size: dim((x_s, f_s)) = (d+1)*N_s
        self.args = args
        # point-net like layers
        if self.args.point_p_net:
            self.point_layer_in  = nn.Conv1d(args.input_dim + 1, args.hidden_dim, 1)
            self.point_layers    = nn.ModuleList()
            for _ in range(args.num_layers):
                self.point_layers.append(nn.Conv1d(args.hidden_dim, args.hidden_dim, 1))
            self.point_layer_afterPooling = nn.Conv1d(args.hidden_dim, args.hidden_dim, 1)
            self.point_layer_out          = nn.Conv1d(args.hidden_dim, u_net_size, 1)
        else:
            self.linearIn = nn.Linear((args.input_dim + 1)*args.N_s, args.hidden_dim)
            self.linear   = nn.ModuleList()
            for _ in range(args.num_layers):
                self.linear.append(nn.Linear(args.hidden_dim, args.hidden_dim))
            self.linearOut = nn.Linear(args.hidden_dim, u_net_size)

        # activation function
        if args.act_p == 'tanh':
            self.act = torch.nn.Tanh()
        elif args.act_p == 'relu':
            self.act = torch.nn.ReLU()
        # elif args.act_p == 'softplus':
        #     if args.tau_sp < 1e-8:
        #         args.tau_sp = np.sqrt(num_params)
        #     self.act = torch.nn.Softplus(beta=args.tau_sp)

    def forward(self, x_s, f_s):
        if self.args.point_p_net:
            x = torch.cat((x_s, f_s), dim=-1).transpose(2,1) # N_f x (d+1) x N_s
            x = self.act(self.point_layer_in(x))             # N_f x h x N_s
            for layer in self.point_layers:
                x = self.act(layer(x))                       # N_f x h x N_s
            
            # global max pooling
            if self.args.pooling == 'max':
                x = torch.max(x, 2, keepdim=True)[0]             # N_f x h x 1
            elif self.args.pooling == 'mean':
                x = torch.mean(x, dim=2).unsqueeze(-1)             # N_f x h x 1
            # one more layer after pooling
            x = self.act(self.point_layer_afterPooling(x))   # N_f x h x 1
            
            return self.point_layer_out(x).squeeze(-1)       # N_f x |theta_u|

        else:
            # x = torch.cat((x_s.view(1,-1), f_s.view(1,-1)), dim=1) # N_f x (d+1)*N_s
            # x = self.linearIn(x)
            x = torch.cat((x_s.view(self.args.N_f,-1), f_s.view(self.args.N_f,-1)), dim=1) # N_f x (d+1)*N_s
            x = self.act(self.linearIn(x))
            for layer in self.linear:
                x = self.act(layer(x))

        return self.linearOut(x)        # N_f x |theta_u|

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


class S_Net(torch.nn.Module):
    def __init__(self, args):
        super(S_Net, self).__init__()

        self.batch_f = args.batch_f

        # u_net: parametrizes the solution
        self.u_net = U_Net_functional(args)
        u_net_size = self.u_net.get_num_params()

        # p_net: predicts the parameters of u_net
        self.p_net = P_Net(args, u_net_size)

        # experiemental
        # self.u_net_fixed_param = U_Net_functional_fixed_param(args)


    def forward(self, x, x_s, f_s):
        if self.batch_f:
            # return torch.stack(
            #     [self.u_net(x_single, self.p_net(x_s_single, f_s_single).squeeze(0)) for 
            #     x_single, x_s_single, f_s_single in 
            #     zip(torch.unbind(x, dim=0), torch.unbind(x_s, dim=0), torch.unbind(f_s, dim=0))
            #     ], dim=0) # N_f x N_int x 1
            params = self.p_net(x_s, f_s) # N_f x |theta_u|
            return torch.stack(
                [self.u_net(x_single, theta) for 
                x_single, theta in 
                zip(torch.unbind(x, dim=0), params)
                ], dim=0) # N_f x N_int x 1   
        else:
            return self.u_net(x, self.p_net(x_s, f_s))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_u_net_param(self, x_s, f_s):
        self.u_net_fixed_param.set_param(self.p_net(x_s, f_s))



class P_Net_MFP(torch.nn.Module):
    def __init__(self, args, u_net_size):
        super(P_Net_MFP, self).__init__()
        self.args = args
        self.linearIn = nn.Linear((args.input_dim + 1)*args.N_s, args.hidden_dim)
        self.linear   = nn.ModuleList()
        for _ in range(args.num_layers):
            self.linear.append(nn.Linear(args.hidden_dim, args.hidden_dim))
        self.linearOut = nn.Linear(args.hidden_dim, u_net_size)

        # activation function
        if args.act_p == 'tanh':
            self.act = torch.nn.Tanh()
        elif args.act_p == 'relu':
            self.act = torch.nn.ReLU()

    def forward(self, theta_f):
        # theta_f: N_f x |theta_u|
        x = self.act(self.linearIn(theta_f))
        for layer in self.linear:
            x = self.act(layer(x))

        return self.linearOut(x)        # N_f x |theta_u|

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())



class S_Net_MFP(torch.nn.Module):
    def __init__(self, args):
        super(S_Net_MFP, self).__init__()

        self.batch_f = args.batch_f

        # u_net: parametrizes the solution
        self.u_net = U_Net_functional(args)
        u_net_size = self.u_net.get_num_params()

        # p_net: predicts the parameters of u_net
        self.p_net = P_Net(args, u_net_size)

        # experiemental
        # self.u_net_fixed_param = U_Net_functional_fixed_param(args)


    def forward(self, x, theta_f):
        if self.batch_f:
            params = self.p_net(theta_f) # N_f x |theta_u|
            return torch.stack(
                [self.u_net(x_single, theta) for 
                x_single, theta in 
                zip(torch.unbind(x, dim=0), params)
                ], dim=0) # N_f x N_int x 1   
        else:
            return self.u_net(x, self.p_net(x_s, f_s))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_u_net_param(self, x_s, f_s):
        self.u_net_fixed_param.set_param(self.p_net(x_s, f_s))



class S_Net_debug(torch.nn.Module):
    def __init__(self, args):
        super(S_Net_debug, self).__init__()

        # u_net: parametrizes the solution
        self.u_net = U_Net_debug(args)
        u_net_size = self.u_net.get_num_params()

        # p_net: predicts the parameters of u_net
        self.p_net = P_Net(args, u_net_size)

        # experiemental
        self.u_net_fixed_param = U_Net_debug(args)


    def forward(self, x, x_s, f_s):
        return self.u_net(x, self.p_net(x_s, f_s))

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def set_u_net_param(self, x_s, f_s):
        self.u_net_fixed_param.set_param(self.p_net(x_s, f_s))


class U_Net_debug(torch.nn.Module):
    def __init__(self, args):
        super(U_Net_debug, self).__init__()
        self.params_size = []
        self.cml_size    = []
        self.args        = args
        self.compute_param_sizes()

    def forward(self, x):
        # input layer
        # x     : (N, *, in_dim)
        # weight: (out_dim, in_dim)
        # bias  : (out_dim)

        # output layer
        return F.tanh(nn.functional.linear(input=x, \
            weight=self.params[:,:self.cml_size[0]].view(1, -1), \
            bias=self.params[:,self.cml_size[0]:].view(-1)))

    
    def compute_param_sizes(self):
        # only 1 layer
        self.params_size.append(self.args.input_dim*1)
        self.params_size.append(1)

        self.cml_size = np.cumsum(self.params_size)
        
    def get_num_params(self):
        return sum(self.params_size)

    def set_param(self, params):
        self.params = params


class U_Net_functional_fixed_param(torch.nn.Module):
    def __init__(self, args):
        super(U_Net_functional_fixed_param, self).__init__()
        self.params_size = []
        self.cml_size    = []
        self.args        = args
        self.compute_param_sizes()

    def forward(self, x):
        # input layer
        # x     : (N, *, in_dim)
        # weight: (out_dim, in_dim)
        # bias  : (out_dim)

        x = F.tanh(nn.functional.linear(input=x, \
            weight=self.params[:,:self.cml_size[0]].view(self.args.hidden_list_u[0], self.args.input_dim), \
            bias=self.params[:,self.cml_size[0]:self.cml_size[1]].view(-1)))
        
        # hidden layers
        for i in range(len(self.args.hidden_list_u)-1):
            x = F.tanh(nn.functional.linear(input=x, \
                weight=self.params[:,self.cml_size[2*i+1]:self.cml_size[2*i+2]].view(self.args.hidden_list_u[i+1], self.args.hidden_list_u[i]), \
                bias=self.params[:,self.cml_size[2*i+2]:self.cml_size[2*i+3]].view(-1)))
        
        # output layer
        return nn.functional.linear(input=x, \
            weight=self.params[:,self.cml_size[-3]:self.cml_size[-2]].view(1, self.args.hidden_list_u[-1]), \
            bias=self.params[:,self.cml_size[-2]:self.cml_size[-1]].view(-1))

    
    def compute_param_sizes(self):
        # input layer, A and b
        self.params_size.append(self.args.input_dim*self.args.hidden_list_u[0])
        self.params_size.append(self.args.hidden_list_u[0])
        # hidden blocks
        for i in range(len(self.args.hidden_list_u)-1):
            self.params_size.append(self.args.hidden_list_u[i]*self.args.hidden_list_u[i+1])
            self.params_size.append(self.args.hidden_list_u[i+1])
        # output layer
        self.params_size.append(self.args.hidden_list_u[-1]*1)
        self.params_size.append(1)

        self.cml_size = np.cumsum(self.params_size)
        
    def get_num_params(self):
        return sum(self.params_size)

    def set_param(self, params):
        self.params = params
        
# class S_Net(torch.nn.Module):
#     def __init__(self, args):
#         super(S_Net, self).__init__()

#         # u_net: parametrizes the solution
#         self.u_net = U_Net(args)
#         # we will not update the weights on u_net as they are outputs of the p_net
#         self.u_net.disable_grad()
#         u_net_size = self.u_net.get_num_params()

#         # p_net: predicts the parameters of u_net
#         self.p_net = P_Net(args, u_net_size)


#     def forward(self, x, x_s, f_s):

#         # forward (x_s, f_s) to get the params for u_net
#         param = self.p_net(x_s, f_s) # 1 x |theta_u|
#         self.u_net.set_params(param)

#         # forward x to get the parametrized u(x)
#         return self.u_net(x)

#     def get_num_params(self):
#         return sum(p.numel() for p in self.parameters() if p.requires_grad)