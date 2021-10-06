import torch
from model import DeepRitzNet, RitzNet, S_Net, S_Net_debug, U_Net_functional, U_Net_vec_functional, P_Net_MFP, \
    U_Net, U_Net_vec
import torch.optim as optim
import numpy as np
import time
import os
import shutil
from torch.utils.data import DataLoader
import scipy.io
import json
from utils import Logger, Plot_logger
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.pyplot import cm
import imageio


class Solver(object):
    def __init__(self, args):
        self.args = args
        # create exp directory
        file = [f for f in args.model_path.split('/')]
        if args.exp_name != None:
            self.experiment_id = args.exp_name
        elif file[-2] == 'models':
            self.experiment_id = file[-3]
        else:
            self.experiment_id = time.strftime('%m%d%H%M%S')
        snapshot_root = 'snapshot/%s' % self.experiment_id
        self.save_dir = os.path.join(snapshot_root, 'models/')
        self.plot_dir = os.path.join(snapshot_root, 'plot_data/')

        # Create directories for logs and pickled models
        if os.path.exists(snapshot_root):
            if args.overwrite_folder:
                shutil.rmtree(snapshot_root, ignore_errors=True)
                os.makedirs(snapshot_root)
            else:
                choose = input("Directory already exists. Remove " + snapshot_root + " ? (y/n)")
                if choose == "y":
                    shutil.rmtree(snapshot_root, ignore_errors=True)
                    os.makedirs(snapshot_root)
        else:
            os.makedirs(snapshot_root)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        
        # initialize loggers
        self.plot_logger = Plot_logger(self.plot_dir)
        sys.stdout       = Logger(os.path.join(snapshot_root, 'log.txt'))

        # sanity checks
        self.check_args()
        
        # pretty print args
        print('input args:\n', json.dumps(vars(args), indent=4, separators=(',', ':')))    


    def init_model(self):
        """Instantiates the model, optimizer, and scheduler

        """

        # grid for the cosine basis
        a          = torch.arange(self.args.M).cuda(self.args.gpu)
        self.coeff_grid       = torch.stack(torch.meshgrid([a]*self.args.input_dim))    # d x M x M x .. x M = d x M^d
        self.coeff_grid_batch = torch.repeat_interleave(self.coeff_grid.unsqueeze(0), self.args.N_f, dim=0)   # N_f x d x M^d

        # grid for x_s, if applicable
        if self.args.xs_sampling == 'grid' or self.args.xs_sampling == 'unif_grid':
            h   = 1 / self.args.num_grid_points_xs
            a   = torch.arange(0, 1+1e-10, step=h).cuda(self.args.gpu)
            x_s = torch.stack(torch.meshgrid([a]*self.args.input_dim), dim=-1).view(-1,self.args.input_dim) # N_s x d
            self.x_s_grid = torch.repeat_interleave(x_s.unsqueeze(0), self.args.N_f, dim=0) # N_f x N_s x d

        if self.args.run_choice == 'poisson_simple' or self.args.run_choice == 'U_Neumann'\
             or self.args.run_choice == 'visualize':
            self.model = RitzNet(self.args)
        elif self.args.run_choice == 'poisson_solution_map' or self.args.run_choice == 'UPS_Neumann'\
            or self.args.run_choice == 'visualize_UPS':
            self.model = S_Net(self.args)
        elif self.args.run_choice == 'solution_map_debug':
            self.model = S_Net_debug(self.args)

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print ('The total number of parameters is: ', total_params)

        if self.args.use_gpu:
            self.model.cuda(self.args.gpu)
        
        # load pretrained model if applicable. 
        # The user is responsible for specifying the model with the correct architecture
        if self.args.model_path != '':
            self.load_pretrain(path=self.args.model_path)

        if self.args.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        if self.args.adaptive_schedule:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                       factor=self.args.lr_decay_gamma, patience=2, verbose=True)
        else:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, \
                step_size=self.args.step_size, gamma=self.args.lr_decay_gamma)    


    def check_args(self):
        """Does several sanity checks to see if the input arg is properly formatted
        """

        if len(self.args.hidden_list_u) == 0:
            raise RuntimeError("Length of hidden layers for the U-net cannot be 0")
        
        if self.args.xs_sampling == 'grid':
            self.args.N_s = (self.args.num_grid_points_xs+1)**self.args.input_dim
        
        if self.args.xs_sampling == 'unif_grid':
            # in this mode, we will sample N_s points from a preset uniform grid each time
            # so we cannot sample more than the total number of grid pts each time.
            self.args.N_s = min((self.args.num_grid_points_xs+1)**self.args.input_dim, self.args.N_s)

        if self.args.run_choice == "MFP_naive" or self.args.run_choice == 'MFP_visualization':
            self.args.input_dim = self.args.d+1
        else:
            self.args.input_dim = self.args.d

        # clean up
        u_list = self.args.hidden_list_u.split(',')
        self.args.hidden_list_u = [int(u) for u in u_list]

    def run(self):
        """Runs the desired function as specified by argparse
        """

        # if self.args.run_choice == 'poisson_simple':
        #     self.poisson_simple()
        # elif self.args.run_choice == 'visualize':
        #     self.visualize_simple_possion()
        # elif self.args.run_choice == 'poisson_simple2':
        #     self.poisson_simple2()
        # elif self.args.run_choice == 'poisson_solution_map':
        #     self.poisson_solution_map()

        # execute the function based on its name
        getattr(self, self.args.run_choice)()
    

    def prepare_data(self, sample_normal=False):
        """Samples data from both the interior and the boundary of the desired domain

        Args:
            

        Returns:
            x_int, x_bdry: sampled data on the interior and the boundary
        """

        # # give a random seed again
        # t = 1000 * time.time() # current time in milliseconds
        # np.random.seed(int(t) % 2**32)

        if self.args.batch_f:
            n_bdry = 0
            # volume is always 1 for the d-dimensional unit cube
            vol    = 1
            # sample from unit cube in R^d
            x_int      = torch.rand(self.args.N_f, self.args.N_int, self.args.input_dim)  # N_f x N_int x d
            # select 1 element each row
            rand_mat   = torch.rand(self.args.N_f, self.args.N_bdry, self.args.input_dim)
            k_th_quant = torch.topk(rand_mat, 1, largest = False)[0][:,:,-1:]
            mask       = rand_mat <= k_th_quant
            # get points on the boundary
            # project to the sides by setting the elements to either 0 or 1
            x_bdry       = torch.rand(self.args.N_f, self.args.N_bdry, self.args.input_dim)
            x_bdry[mask] = torch.round(torch.rand(x_bdry[mask].shape))
        else:
            n_bdry = 0
            # volume is always 1 for the d-dimensional unit cube
            vol    = 1
            # sample from unit cube in R^d
            x_int      = torch.rand(self.args.N_int, self.args.input_dim)   # N_int x d
            # select 1 element each row
            rand_mat   = torch.rand(self.args.N_bdry, self.args.input_dim)  # N_int x d
            k_th_quant = torch.topk(rand_mat, 1, largest = False)[0][:,-1:] # N_int x d
            mask       = rand_mat <= k_th_quant
            # get points on the boundary
            # project to the sides by setting the elements to either 0 or 1
            x_bdry       = torch.rand(self.args.N_bdry, self.args.input_dim)
            x_bdry[mask] = torch.round(torch.rand(x_bdry[mask].shape))

            # sample normal vectors on the bdry if desired (for Neumann BC)
            if sample_normal:
                pass
                
            # TODO: implement more regions

        return x_int.cuda(self.args.gpu), x_bdry.cuda(self.args.gpu), n_bdry, vol


    def forcing(self, x):
        """calculates the value of the forcing term with input x

        Args:
            x (tensor): B x d

        Returns:
            tensor: B x 1
        """
        # for poisson_simple
        # assume each row is a data point
        return torch.sum(torch.sin(np.pi/2*x), dim=1)

    def dudx_exact(self, x):
        return np.pi/2* torch.cos(np.pi/2 *x)


    def prepare_f(self):
        if self.args.batch_f:
            self.b = self.args.b_mult * (2 * torch.rand([self.args.N_f] \
                + [self.args.M]*self.args.input_dim).cuda(self.args.gpu) -1 )         # N_f x M^d
            self.a = self.b / (np.pi**2 * torch.sum(self.coeff_grid_batch**2, dim=1)) # N_f x M^d
            # compatiblity condition
            # below does: b[:, 0,0, ... 0] = 0
            self.b[(slice(None),) + (0,) * self.args.input_dim] = 0
            self.a[(slice(None),) + (0,) * self.args.input_dim] = 0
        else:
            # each b_m is in [-1,1]
            self.b   = self.args.b_mult * \
                (2 * torch.rand([self.args.M]*self.args.input_dim).cuda(self.args.gpu) - 1) # M x M x ... x M = M^d
            self.a   = self.b / (np.pi**2 * torch.sum(self.coeff_grid**2, dim=0))           # M^d
            # compatiblity condition
            self.b[tuple([0]*self.args.input_dim)] = 0
            self.a[tuple([0]*self.args.input_dim)] = 0


    def exact_soln(self, x):
        """calculates the value of the exact solution on the points x

        Args:
            x (tensor): N x d

        Returns:
            tensor: N x 1
        """

        if self.args.run_choice == 'poisson_simple':
            return torch.sum(torch.sin(np.pi/2*x), dim=1)
        elif self.args.run_choice == 'poisson_solution_map' or self.args.run_choice == 'UPS_Neumann'\
            or self.args.run_choice == 'U_Neumann' or self.args.run_choice == 'visualize_UPS':
            if self.args.batch_f:
                COS = torch.cos(torch.einsum('md..., mnd -> m...nd', np.pi*self.coeff_grid_batch, x))  # N_f x M^d x N x d
                C   = torch.prod(COS, dim=-1)                                                          # N_f x M^d x N
                A   = torch.repeat_interleave(self.a.unsqueeze(-1), x.shape[1], dim=-1)                # N_f x M^d x N

                return torch.sum(A*C, dim=[i+1 for i in range(self.args.input_dim)]).unsqueeze(-1)      # N_f x N x 1
            else:
                COS = torch.cos(torch.einsum('k...,nk -> ...nk', np.pi*self.coeff_grid, x))     # M^d x N x d
                C   = torch.prod(COS, dim=-1)                                                   # M^d x N
                A   = torch.repeat_interleave(self.a.unsqueeze(-1), x.shape[0], dim=-1)         # M^d x N

                return torch.sum(A*C, dim=[i for i in range(self.args.input_dim)]).unsqueeze(1) # N x 1
        elif self.args.run_choice == 'MFP_naive' and self.args.MFP_ICTC == 'gaussian':
            return 1/4 * self.args.d
        else:
            raise NotImplementedError("Exact solution not available for this run mode.")


    def obj_min(self):
        """Returns the theoretical minimum of the loss function

        Raises:
            NotImplementedError: The analytic expression is not always available.

        Returns:
            float: minimal Ritz energy
        """

        if self.args.run_choice == 'U_Neumann':
            # this is just a single f
            # return -1/8 * torch.sum(self.b * self.a)
            # adjust a & b so that the entries are weighted according to the # of zeros in their indices
            weight = 2.**(-torch.sum(self.coeff_grid != 0, dim=0))
            return -1/2 * torch.sum(self.a * self.b * weight)
        elif self.args.run_choice == 'UPS_Neumann':
            # self.b: N_f x M^d
            weight = 2.**(-torch.sum(self.coeff_grid_batch != 0, dim=1))
            if self.args.batch_f:
                return -1/2 * torch.sum(self.a * self.b * weight) / self.args.N_f
            else:
                return -1/2 * torch.sum(self.a * self.b * weight)
        else:
            raise NotImplementedError("Theoretical value of the objective not available.")


    def f(self, x):
        """Evaluate f(x)

        Args:
            x (tensor): N x d / N_f x N x d (batch)

        Returns:
            value of f on x: N x 1
        """
        if self.args.run_choice == 'poisson_simple':
            return torch.sum(torch.sin(np.pi/2*x), dim=1).unsqueeze(1)
        elif self.args.run_choice == 'poisson_solution_map' or self.args.run_choice == 'UPS_Neumann' \
            or self.args.run_choice == 'U_Neumann' or self.args.run_choice == 'solution_map_debug' or \
                self.args.run_choice == 'visualize_UPS':
            if self.args.batch_f:
                COS = torch.cos(torch.einsum('md..., mnd -> m...nd', np.pi*self.coeff_grid_batch, x))  # N_f x M^d x N x d
                C   = torch.prod(COS, dim=-1)                                                          # N_f x M^d x N
                B   = torch.repeat_interleave(self.b.unsqueeze(-1), x.shape[1], dim=-1)                # N_f x M^d x N

                return torch.sum(B*C, dim=[i+1 for i in range(self.args.input_dim)]).unsqueeze(-1)      # N_f x N x 1
            else:
                COS = torch.cos(torch.einsum('k..., nk -> ...nk', np.pi*self.coeff_grid, x))    # M^d x N x d
                C   = torch.prod(COS, dim=-1)                                                   # M^d x N
                B   = torch.repeat_interleave(self.b.unsqueeze(-1), x.shape[0], dim=-1)         # M^d x N

                return torch.sum(B*C, dim=[i for i in range(self.args.input_dim)]).unsqueeze(1) # N x 1
        else:
            raise NotImplementedError("Cannot compute f in this run mode.")


    def f_debug(self, x):
        """Evaluate f at x in the slowest way possible, but it's definitely correct.

        Args:
            x (tensor): N x d
        """
        if self.args.batch_f:
            S = torch.zeros((self.args.N_f, x.shape[1], 1)).cuda(self.args.gpu)
            for b in range(self.args.N_f):
                for n in range(x.shape[1]):
                    for i in range(self.b.shape[1]):
                        for j in range(self.b.shape[2]):
                            S[b,n,0] += self.b[b,i,j] * torch.cos(i*np.pi*x[b,n,0]) * torch.cos(j*np.pi*x[b,n,1])
        else:
            S = torch.zeros((x.shape[0], 1)).cuda(self.args.gpu)
            for n in range(x.shape[0]):
                for i in range(self.b.shape[0]):
                    for j in range(self.b.shape[1]):
                        S[n] += self.b[i,j] * torch.cos(i*np.pi*x[n,0]) * torch.cos(j*np.pi*x[n,1])
        
        return S


    def prepare_xs(self):
        # sample from unit cube in R^d
        if self.args.batch_f:
            if self.args.xs_sampling == 'grid':
                # For this mode, we use a fixed, preset grid
                return self.x_s_grid
            elif self.args.xs_sampling == 'unif':
                # For this mode, we sample the points uniformly from the domain and order them by norm
                if self.args.same_batch_xs:
                    x_s  = torch.rand(self.args.N_s, self.args.input_dim).cuda(self.args.gpu) # N_s x d
                    # order the points radially
                    norm = torch.norm(x_s, dim=-1)
                    idx  = torch.sort(norm)[1]
                    x_s  = x_s[idx, :]
                    # repeat along N_f to use the same x_s for the batch of functions
                    return torch.repeat_interleave(x_s.unsqueeze(0), self.args.N_f, dim=0)    # N_f x N_s x d
                else:
                    x_s = torch.rand(self.args.N_f, self.args.N_s, self.args.input_dim).cuda(self.args.gpu)  # N_f x N_s x d
                    # order the points radially
                    norm = torch.norm(x_s, dim=-1)
                    idx  = torch.sort(norm)[1]
                    I = torch.repeat_interleave(idx.unsqueeze(-1), 2, dim=-1)
                    # basically batch sort
                    return torch.gather(x_s,1,I) # N_f x N_s x d
            elif self.args.xs_sampling == 'unif_grid':
                # For this mode, we sample N_s points from a fixed, preset grid
                if self.args.same_batch_xs:
                    # sample from a preset grid
                    grid = self.x_s_grid[0]
                    # uniformly sample points from grid in sorted order
                    # e.g. if the grid is 0 to 1 in 0.1 increments, one sample can be: 0.3, 0.5, 0.6, 0.9
                    # idx  = torch.from_numpy(np.sort(np.random.choice(grid.shape[0], \
                    #     size=self.args.N_s, replace=False))).long().cuda(self.args.gpu)
                    # return torch.repeat_interleave(grid[idx].unsqueeze(0), self.args.N_f, dim=0) # N_f x N_s x d
                    idx  = torch.from_numpy(np.random.choice(grid.shape[0], \
                        size=self.args.N_s, replace=False)).long().cuda(self.args.gpu)
                    return torch.repeat_interleave(grid[idx].unsqueeze(0), self.args.N_f, dim=0) # N_f x N_s x d
                else:
                    raise NotImplementedError("Not Implemented")
        else:
            if self.args.xs_sampling == 'grid':
                raise NotImplementedError("Not Implemented")
            else:
                return torch.rand(self.args.N_s, self.args.input_dim).cuda(self.args.gpu)  # N_int x d


    # =================================================================================== #
    #                                          MFP                                        #
    # =================================================================================== #

    def init_model_MFP(self):
        self.rho   = U_Net(self.args)
        self.m     = U_Net_vec(self.args)

        rho_params = sum(p.numel() for p in self.rho.parameters() if p.requires_grad)
        m_params   = sum(p.numel() for p in self.m.parameters() if p.requires_grad)

        print ('Number of parameters in rho net: ', rho_params)
        print ('Number of parameters in m net: ', m_params)

        # set various flags to detect if there's a bug
        # if self.args.debug:
        torch.autograd.set_detect_anomaly(True)

        if self.args.use_gpu:
            self.rho.cuda(self.args.gpu)
            self.m.cuda(self.args.gpu)
        
        if self.args.rho_path != '' and self.args.m_path != '':
            self.load_pretrain(rho_path=self.args.rho_path, m_path=self.args.m_path, mode='MFP')

        if self.args.optimizer == 'adam':
            self.optimizer_rho = optim.Adam(self.rho.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            self.optimizer_m   = optim.Adam(self.m.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'sgd':
            self.optimizer_rho = optim.SGD(self.rho.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            self.optimizer_m   = optim.SGD(self.m.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        if self.args.adaptive_schedule:
            self.scheduler_rho = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_rho, mode='min',
                                                       factor=self.args.lr_decay_gamma, patience=2, verbose=True)
            self.scheduler_m   = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_m, mode='min',
                                                       factor=self.args.lr_decay_gamma, patience=2, verbose=True)
        else:
            self.scheduler_rho = optim.lr_scheduler.StepLR(self.optimizer_rho, \
                step_size=self.args.step_size, gamma=self.args.lr_decay_gamma)
            self.scheduler_m   = optim.lr_scheduler.StepLR(self.optimizer_m, \
                step_size=self.args.step_size, gamma=self.args.lr_decay_gamma)
        
        if self.args.MFP_ICTC == 'gaussian':
            self.gaussian_0 = torch.distributions.multivariate_normal.MultivariateNormal(torch.tensor([1/4] * self.args.d).cuda(self.args.gpu), \
                1/50*torch.eye(self.args.d).cuda(self.args.gpu))
            self.gaussian_1 = torch.distributions.multivariate_normal.MultivariateNormal(torch.tensor([3/4] * self.args.d).cuda(self.args.gpu), \
                 1/50*torch.eye(self.args.d).cuda(self.args.gpu))


    def sample_data_MFP(self):
        if self.args.MFP_sampling == 'sqr':
            # time
            t          = torch.rand(self.args.N_int, 1).cuda(self.args.gpu)                  # N_int x 1
            t_int_sqr  = torch.repeat_interleave(t, self.args.N_int, dim=0)                  # N_int^2 x 1
            t_bdry_sqr = torch.repeat_interleave(t, self.args.N_bdry, dim=0)                 # N_bdry*N_int x 1
            t_0        = torch.zeros(self.args.N_int,1).cuda(self.args.gpu)
            t_1        = torch.ones(self.args.N_int,1).cuda(self.args.gpu)      

            # x_int
            x_int      = torch.rand(self.args.N_int, self.args.d).cuda(self.args.gpu)        # N_int x d
            x_int_sqr  = x_int.repeat(self.args.N_int,1)                                     # N_int^2 x d
            
            # x_bdry
            # masks for projection
            rand_mat   = torch.rand(self.args.N_bdry, self.args.d).cuda(self.args.gpu)
            k_th_quant = torch.topk(rand_mat, 1, largest = False)[0][:,-1:]
            mask       = rand_mat <= k_th_quant
            # get the actual points 
            x_bdry       = torch.rand(self.args.N_bdry, self.args.d).cuda(self.args.gpu)
            x_bdry[mask] = torch.round(torch.rand(x_bdry[mask].shape).cuda(self.args.gpu))   # N_bdry x d
            x_bdry_sqr   = x_bdry.repeat(self.args.N_int,1)                                  # N_bdry*N_int x d
            # get the outward pointing normals associated with x_bdry
            n_bdry_sqr = torch.zeros_like(x_bdry_sqr)                       # N_bdry*N_int x d
            n_bdry_sqr[x_bdry_sqr == 0] = -1
            n_bdry_sqr[x_bdry_sqr == 1] = 1

            return t, x_int, x_bdry, x_int_sqr, t_int_sqr, t_bdry_sqr, x_bdry_sqr, n_bdry_sqr, t_0, t_1
            
        elif self.args.MFP_sampling == 'nonsqr':
            # time
            t          = torch.rand(self.args.N_int, 1).cuda(self.args.gpu)                  # N_int x 1
            t_int_sqr  = t                                                                   # N_int x 1
            t_bdry     = torch.rand(self.args.N_bdry, 1).cuda(self.args.gpu)                 # N_bdry x 1
            t_0        = torch.zeros(self.args.N_int,1).cuda(self.args.gpu)                  # N_int x 1
            t_1        = torch.ones(self.args.N_int,1).cuda(self.args.gpu)                   # N_int x 1

            # x_int
            x_int      = torch.rand(self.args.N_int, self.args.d).cuda(self.args.gpu)        # N_int x d                                                            # N_int x d
            
            # x_bdry
            # masks for projection
            rand_mat   = torch.rand(self.args.N_bdry, self.args.d).cuda(self.args.gpu)
            k_th_quant = torch.topk(rand_mat, 1, largest = False)[0][:,-1:]
            mask       = rand_mat <= k_th_quant
            # get the actual points 
            x_bdry       = torch.rand(self.args.N_bdry, self.args.d).cuda(self.args.gpu)
            x_bdry[mask] = torch.round(torch.rand(x_bdry[mask].shape).cuda(self.args.gpu))   # N_bdry x d
            # x_bdry_sqr   = x_bdry.repeat(self.args.N_int,1)                                  # N_bdry*N_int x d
            # get the outward pointing normals associated with x_bdry
            n_bdry = torch.zeros_like(x_bdry)                       # N_bdry x d
            n_bdry[x_bdry == 0] = -1
            n_bdry[x_bdry == 1] = 1

            return t, x_int, x_bdry, x_int, t, t_bdry, x_bdry, n_bdry, t_0, t_1

    def sigmoid(self, a, b, c, x):
        """Implements the sigmoid function: b * 1 / (1 + exp(-a*(x-c)))

        Args:
            a (float): slope
            b (float): amplitude (where the function asympotes)
            c (float): shift to the right
            x (float): input

        Returns:
            output of the modified sigmoid function
        """
        return b / (1 + np.exp(-a*(x-c)))

    def warm_up(self, lbd, iter_num):
        """Warm up the penalty weight by a sigmoid function.
        The way it works: the weight starts at lbd_init and reaches p*lbd half way through training.
        The plateauing value is lbd.

        Args:
            lbd (float): max value of the penalty term

        Returns:
            The warmed up value of the penalty term.
        """
        a = 2 * (np.log(lbd/self.args.lbd_init - 1) - np.log(1/self.args.p-1)) / self.args.num_iter
        c = np.log(lbd/self.args.lbd_init-1) / a

        return self.sigmoid(a, lbd, c, iter_num)

    def rho_0(self, x):
        # uniform over the domain
        if self.args.MFP_ICTC == 'sqr_circ':
            return torch.ones(x.shape[0], 1).cuda(self.args.gpu)
        elif self.args.MFP_ICTC == 'gaussian':
            return torch.exp(self.gaussian_0.log_prob(x)).reshape(-1,1)

    def rho_1(self, x):
        if self.args.MFP_ICTC == 'sqr_circ':
            # uniform over the unit ball 
            a = torch.exp(torch.lgamma(torch.as_tensor(self.args.d/2 + 1))) / np.pi**(self.args.d/2) / (1/2)**self.args.d
            mask = (torch.norm(x, dim=-1) < 1).cuda(self.args.gpu)

            return torch.zeros(x.shape[0], 1).cuda(self.args.gpu).masked_fill_(mask.unsqueeze(-1), a)
        elif self.args.MFP_ICTC == 'gaussian':
             return torch.exp(self.gaussian_1.log_prob(x)).reshape(-1,1)


    def test_loss(self, t_batch, x_batch):
        s = 0
        for t in t_batch:
            for x in x_batch:
                s += torch.norm(self.m(t,x))**2 / self.rho(t,x)

        return s / len(t_batch) / len(x_batch)    
        
    def MFP_visualization(self):
        if self.args.rho_path == '' or self.args.m_path == '':
            raise RuntimeError("Has to load both rho and m net for visualization.")

        self.init_model_MFP()
        
        # construct 2d grid
        grid_1D = torch.arange(0,1+self.args.mesh_size, step=self.args.mesh_size)
        grid    = torch.meshgrid(grid_1D, grid_1D)                     # a tuple of N x N tensors
        grid_2D = torch.stack(grid, dim=-1)                            # N x N x 2
        N       = grid_2D.shape[0]
        grid    = grid_2D.view(-1,2).cuda(self.args.gpu)               # N^2 x 2

        self.rho.eval()

        # create the figure
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(np.random.random((N,N)), interpolation='nearest', cmap=cm.Blues)
        plt.show(block=False)
        t  = 0
        i  = 0
        dt = 0.01
        
        # draw some data in loop
        while t < 1:
            # wait
            time.sleep(dt)
            t_vec = torch.zeros(grid.shape[0],1).fill_(t).cuda(self.args.gpu)
            val   = self.rho(t_vec, grid)
            val   = val.detach().cpu().numpy().reshape(N,N)
            # replace the image contents
            im.set_array(val)
            # redraw the figure
            fig.canvas.draw()
            plt.savefig('movies/fig_' + str(i) + '.png')
            t += dt
            i += 1

        plt.show()

    def make_movie(self):
        f_length = 100
        filenames = ['movies/fig_' + str(i) + '.png' for i in range(f_length)]
        images = []
        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave('movies/gaussian_2d.gif', images)

    def MFP_naive(self):
        self.init_model_MFP()
        loss_sum       = 0
        L_cost_gap_sum = 0     
        L_cost_sum     = 0
        L_IC_sum       = 0
        L_TC_sum       = 0
        L_conserv_sum  = 0

        # training loop
        for i in range(self.args.num_iter):
            # sample points as data
            if i % self.args.resample_step_x == 0:
                # t:   N_x x 1
                # x:   N_x x d
                # t_x: N_x^2 x (d+1)
                # x_0: N_x x (d+1)
                # x_1: N_x x (d+1)
                # _, x, t_sqr, x_sqr, _, t_0, t_1 = self.sample_data_MFP()
                t, x_int, x_bdry, x_int_sqr, t_int_sqr, t_bdry_sqr, x_bdry_sqr, \
                    n_bdry_sqr, t_0, t_1 = self.sample_data_MFP()
            
            self.rho.train()
            self.rho.zero_grad()
            self.m.train()
            self.m.zero_grad()

            # set this flag before forwarding it through the model
            t_int_sqr.requires_grad = True
            x_int_sqr.requires_grad = True
            rho_out = self.rho(t_int_sqr, x_int_sqr) # N_int x 1
            m_out   = self.m(t_int_sqr, x_int_sqr)   # N_int x d
            m_bdry  = self.m(t_bdry_sqr, x_bdry_sqr) # N_bdry x d

            drho_dt = torch.autograd.grad(rho_out, t_int_sqr, \
                 grad_outputs=torch.ones_like(rho_out), \
                retain_graph=True,create_graph=True,only_inputs=True)[0]                     # N_int^2 x 1

            ##### This may not be the most optimized way to compute div(m) b/c there's kind of a loop here ######
            # Below computes: dm[:,i] = dx_i (m_i)
            dm      = torch.stack([ 
                torch.autograd.grad(m_i, x_int_sqr, grad_outputs=torch.ones_like(m_i), \
                retain_graph=True,create_graph=True,only_inputs=True)[0][:,i]
                for i, m_i in enumerate(torch.unbind(m_out, dim=-1))
            ], dim=-1)  # N_int^2 x d

            div_x_m = torch.sum(dm, dim=-1, keepdim=True) # N_int^2 x 1

            # loss
            # L_cost equal to: 1/self.args.N_int**2 * torch.norm(m_out / torch.sqrt(rho_out))**2
            if self.args.MFP_sampling == 'sqr':
                L_cost     = 1/self.args.N_int**2 * torch.sum(m_out**2 / rho_out)
                L_IC       = 1/self.args.N_int * torch.norm(self.rho(t_0, x_int) - self.rho_0(x_int))
                L_TC       = 1/self.args.N_int * torch.norm(self.rho(t_1, x_int) - self.rho_1(x_int))
                # L_conserv  = 1/self.args.N_int**2 * torch.norm(drho_dt - div_x_m)**2
                L_conserv  = 1/self.args.N_int**2 * torch.norm(drho_dt - div_x_m) + \
                    1/self.args.N_bdry/self.args.N_int * torch.norm(torch.sum(m_bdry * n_bdry_sqr, dim=-1, keepdim=True))
            elif self.args.MFP_sampling == 'nonsqr':
                # L_cost     = 1/self.args.N_int**(1/2) * torch.sum(m_out**2 / rho_out)
                # L_IC       = 1/self.args.N_int**(1/2) * torch.norm(self.rho(t_0, x_int) - self.rho_0(x_int))
                # L_TC       = 1/self.args.N_int**(1/2) * torch.norm(self.rho(t_1, x_int) - self.rho_1(x_int))
                # L_conserv  = 1/self.args.N_int**(1/2) * torch.norm(drho_dt - div_x_m) + \
                #     1/self.args.N_bdry**(1/2) * torch.norm(torch.sum(m_bdry * n_bdry_sqr, dim=-1, keepdim=True))
                
                L_cost     = 1/self.args.N_int * torch.sum(m_out**2 / rho_out)
                L_IC       = 1/self.args.N_int * torch.norm(self.rho(t_0, x_int) - self.rho_0(x_int))**2
                L_TC       = 1/self.args.N_int * torch.norm(self.rho(t_1, x_int) - self.rho_1(x_int))**2
                L_conserv  = 1/self.args.N_int * torch.norm(drho_dt - div_x_m)**2 + \
                    1/self.args.N_bdry * torch.norm(torch.sum(m_bdry * n_bdry_sqr, dim=-1, keepdim=True))**2
            
            # warm up for penalty lbd's
            if self.args.penalty_warmup == 'none':
                lbd_ICTC    = self.args.lbd_1
                lbd_conserv = self.args.lbd_2
            elif self.args.penalty_warmup == 'ICBC':
                lbd_ICTC    = self.warm_up(self.args.lbd_1, i)
                lbd_conserv = self.args.lbd_2
            elif self.args.penalty_warmup == 'conserv':
                lbd_ICTC    = self.args.lbd_1
                lbd_conserv = self.warm_up(self.args.lbd_2, i)
            elif self.args.penalty_warmup == 'all':
                lbd_ICTC    = self.warm_up(self.args.lbd_1, i)
                lbd_conserv = self.warm_up(self.args.lbd_2, i)

            loss = L_cost + lbd_ICTC * (L_IC + L_TC) + lbd_conserv * L_conserv
            loss.backward()

            # debug
            if self.args.debug:
                print('Norm of drhodt: {}'.format(torch.norm(drho_dt).item()))
                print('Norm of div_x_m: {}'.format(torch.norm(div_x_m).item()))
                print('Norm of m_out: {}'.format(torch.norm(m_out).item()))
                print('Norm of rho_out: {}'.format(torch.norm(rho_out).item()))
                print('Norm of m_bdry: {}'.format(torch.norm(m_bdry).item()))
                print('Norm of rho at t=0: {}'.format(torch.norm(self.rho(t_0, x_int)).item()))
                print('Norm of rho at t=1: {}'.format(torch.norm(self.rho(t_1, x_int)).item()))
                print('Norm of true rho at t=0: {}'.format(torch.norm(self.rho_0(x_int)).item()))
                print('Norm of true rho at t=1: {}'.format(torch.norm(self.rho_1(x_int)).item()))
                # print("rho net grad norm: {:.2f}, m net grad norm: {:.2f}".format(\
                #     sum([torch.norm(p.grad.data)**2 for p in self.rho.parameters()]), \
                #         sum([torch.norm(p.grad.data)**2 for p in self.m.parameters()]) ) )
                # print("rho net weight norm: {:.2f}, m net weight norm: {:.2f}".format(\
                #     sum([torch.norm(p.data)**2 for p in self.rho.parameters()]), \
                #         sum([torch.norm(p.data)**2 for p in self.m.parameters()]) ) )
                print("rho net grad norm: {:.2f}".format(\
                    sum([torch.norm(p.grad.data)**2 for p in self.rho.parameters()]) ) )
                print("rho net weight norm: {:.2f}".format(\
                    sum([torch.norm(p.data)**2 for p in self.rho.parameters()])) )
                print("Loss: {:.4f}, L_cost:{:.4f}, L_IC:{:.4f}, L_TC:{:.4f}, L_conserv:{:.4f}".format(\
                    loss, L_cost, L_IC, L_TC, L_conserv))
                print('\n')

            self.optimizer_rho.step()
            self.optimizer_m.step()

            # recording history
            loss_sum      += float(loss)
            if self.args.MFP_ICTC == 'gaussian':
                exact_cost       = self.exact_soln(x_int)
                loss_gap         = float(torch.abs(L_cost - exact_cost))
                L_cost_gap_sum  += loss_gap
            elif self.args.MFP_ICTC == 'sqr_circ':
                # no analytic expression for exact cost
                exact_cost       = 0

            L_cost_sum    += float(L_cost)
            L_IC_sum      += float(L_IC)
            L_TC_sum      += float(L_TC)
            L_conserv_sum += float(L_conserv)

            # logging training history
            if (i+1) % self.args.log_step == 0:

                # average error over an 'epoch'
                epoch          = (self.args.start_iter + i+1) // self.args.log_step
                loss_avg       = float(loss_sum) / self.args.log_step
                L_cost_gap_avg = float(L_cost_gap_sum) / self.args.log_step
                L_cost_avg     = float(L_cost_sum) / self.args.log_step
                L_IC_avg       = float(L_IC_sum) / self.args.log_step
                L_TC_avg       = float(L_TC_sum) / self.args.log_step
                L_conserv_avg  = float(L_conserv_sum) / self.args.log_step

                print("Loss: {:.4f}, L_cost:{:.4f}, L_cost_thry:{:.4f}, L_cost_gap:{:.4f}, L_IC:{:.4f}, L_TC:{:.4f}, L_conserv:{:.4f}".format(\
                    loss_avg, L_cost_avg, exact_cost, L_cost_gap_avg, L_IC_avg, L_TC_avg, L_conserv_avg))
                print("LR: {}".format(str(self.optimizer_rho.param_groups[0]['lr'])))
                print("lbd ICTC: {:.4f}, lbd conserv: {:.4f}".format(lbd_ICTC, lbd_conserv))
                
                # save data to txt
                self.plot_logger.log('iter_num', self.args.start_iter + i+1)
                self.plot_logger.log('epoch', epoch)
                self.plot_logger.log('loss', loss_avg)
                self.plot_logger.log('l_IC', L_IC_avg)
                self.plot_logger.log('l_TC', L_TC_avg)
                self.plot_logger.log('l_cost', L_cost_avg)
                self.plot_logger.log('l_conserv', L_conserv_avg)
                
                # reset history containers
                loss_sum       = 0
                L_cost_gap_sum = 0
                L_cost_sum     = 0
                L_IC_sum       = 0
                L_TC_sum       = 0
                L_conserv_sum  = 0

            # LR decay
            if self.args.adaptive_schedule:
                self.scheduler_rho.step(loss)
                self.scheduler_m.step(loss)
            else:
                self.scheduler_rho.step()
                self.scheduler_m.step()

            # save model
            if (i + 1) % self.args.snapshot_interval == 0:
                    self.snapshot(self.args.start_iter + i + 1, mode='MFP')

        # save model
        self.snapshot(self.args.start_iter + self.args.num_iter, mode='MFP')


    # =================================================================================== #
    #                           UPS-Net for the solution map                              #
    # =================================================================================== #

    def UPS_Neumann(self):
        self.init_model()

        # # construct 2d grid
        grid_oneDim = torch.arange(0, 1+1e-10, step=self.args.mesh_size)
        grid        = torch.meshgrid(grid_oneDim, grid_oneDim)             # a tuple of N x N tensors
        grid_2d     = torch.stack(grid, dim=-1)                            # N x N x 2
        N           = grid_2d.shape[0]
        grid        = grid_2d.view(-1,2).cuda(self.args.gpu)
        if self.args.batch_f:
            grid = torch.repeat_interleave(grid.unsqueeze(0), self.args.N_f, dim=0)

        # containers
        loss_sum  = 0
        e_inf_sum = 0
        e_1_sum   = 0
        e_rel_sum = 0

        # training loop
        for i in range(self.args.num_iter):
            # there are three things we have to sample: f, x_s, x_i / x_b
            # The resample steps should be chosen s.t. step_x = a*step_xs = a*b*step_f
            # e.g. step_x = 10, step_xs = 10, step_f = 100
            if i % self.args.resample_step_f == 0:
                # print ("Resampled f")
                self.prepare_f()

            if i % self.args.resample_step_xs == 0 and not self.args.same_x:
                # print ("Resampled x_s")
                x_s = self.prepare_xs() # N_f x N_s x d
                f_s = self.f(x_s)       # N_f x N_s x 1

            if i % self.args.resample_step_x == 0:
                # print ("Resampled x")
                x_int, _, _, vol = self.prepare_data()  # N_f x N_int x d
                f_int            = self.f(x_int)        # N_f x N_int x 1
                if self.args.same_x:
                    x_s = x_int.clone()
                    f_s = f_int.clone()

            self.model.train()
            self.optimizer.zero_grad()

            # set this flag before forwarding it through the model
            x_int.requires_grad = True
            # get u(x;theta) on int
            u_int  = self.model(x_int, x_s, f_s)  # N_f x N_int x 1

            # compute grad_x(u): N_f x N_int x d
            dSdx = torch.autograd.grad(u_int, x_int, grad_outputs=torch.ones_like(u_int), \
                retain_graph=True,create_graph=True,only_inputs=True)[0]

            # loss on the interior
            if self.args.batch_f:
                L_int = vol/self.args.N_int/self.args.N_f * (1/2*torch.norm(dSdx)**2 - torch.sum(u_int * f_int)) + \
                    vol**2/(2*self.args.N_f*self.args.N_int**2) * torch.sum(torch.sum(u_int, dim=1)**2)
            else:
                L_int = vol/self.args.N_int * (1/2*torch.norm(dSdx)**2 - torch.sum(u_int * f_int)) + \
                    1/2*(vol/self.args.N_int * torch.sum(u_int))**2

            # No loss on the boundary

            # overall loss
            loss = L_int
            loss.backward()
            self.optimizer.step()

            # recording history
            loss_sum += float(loss)
            self.model.eval()
            with torch.no_grad():
                u_nn      = self.model(grid, x_s, f_s)    # N_f x N^2 x 1
                u_exact   = self.exact_soln(grid)         # N_f x N^2 x 1
            # L_inf error
            diff       = torch.abs(u_nn - u_exact)
            # e_inf      = torch.max(diff)
            e_inf      = torch.mean(torch.max(diff.squeeze(-1), dim=-1)[0])
            e_1_mean   = torch.mean(diff)
            e_rel      = torch.mean(torch.norm(u_nn.squeeze(-1)-u_exact.squeeze(-1), dim=-1) / torch.norm(u_exact.squeeze(-1), dim=-1))
            e_1_sum   += float(e_1_mean)
            e_inf_sum += float(e_inf)
            e_rel_sum += float(e_rel)

            # logging training history
            if (i+1) % self.args.log_step == 0:
                # averaging
                epoch     = (self.args.start_iter+i+1) // self.args.log_step
                # if self.args.batch_f:
                #     e_1_mean = float(e_1_sum) / self.args.log_step / N**2 / self.args.N_f
                # else:
                #     e_1_mean = float(e_1_sum) / self.args.log_step / N**2
                e_1_mean  = float(e_1_sum) / self.args.log_step
                e_inf     = float(e_inf_sum) / self.args.log_step
                e_rel     = float(e_rel_sum) / self.args.log_step
                loss_mean = float(loss_sum) / self.args.log_step
                loss_thry = self.obj_min()

                print ("Loss value: {:.4f}, Theoretical min: {:.4f}".format(loss_mean, loss_thry))
                print ("Model weight norm: {:.2f}".format(sum([torch.norm(p.data)**2 for p in self.model.parameters()] )))
                print ("Grad norm: {:.2f}".format(sum([torch.norm(p.grad.data)**2 for p in self.model.parameters()] )))
                print ("Mean L_inf error: {:.4f}, Mean Abs. error: {:.4f}, Relative L_2 error:{:.4f}".format(e_inf, e_1_mean, e_rel))
                # self.model.set_u_net_param(x_s, f_s)
                # L = - torch.stack([
                #     torch.trace(torch.autograd.functional.hessian(self.model.u_net_fixed_param, x_i))
                #     for x_i in torch.unbind(x_int, dim=0)
                #     ], dim=0).reshape(-1,1)
                # residual = torch.max(torch.abs(L - f_int))
                # print("|-Delta u - f|: {:.4f}".format(residual.item()))

                print("LR: {}".format(str(self.optimizer.param_groups[0]['lr'])))

                # save data to txt
                self.plot_logger.log('iter_num', self.args.start_iter+ i+1)
                self.plot_logger.log('epoch', epoch)
                self.plot_logger.log('loss', loss_mean)
                self.plot_logger.log('loss_thry', float(loss_thry))
                self.plot_logger.log('e_inf', e_inf)
                self.plot_logger.log('e_1', e_1_mean)
                self.plot_logger.log('e_rel', e_rel)
                
                # reset history containers
                loss_sum  = 0
                e_inf_sum = 0
                e_1_sum   = 0
                e_rel_sum = 0

            # LR decay
            if self.args.adaptive_schedule:
                self.scheduler.step(loss)
            else:
                self.scheduler.step()

            # save model
            if (i + 1) % self.args.snapshot_interval == 0:
                    self.snapshot(self.args.start_iter + i + 1)

        # save model
        self.snapshot(self.args.start_iter + self.args.num_iter)

    
    def poisson_solution_map(self):
        self.init_model()

        # # construct 2d grid
        # grid_oneDim = torch.arange(0, 1+self.args.mesh_size, step=self.args.mesh_size)
        # grid        = torch.meshgrid(grid_oneDim, grid_oneDim)             # a tuple of N x N tensors
        # grid_2d     = torch.stack(grid, dim=-1)                            # N x N x 2
        # N           = grid_2d.shape[0]
        # grid        = grid_2d.view(-1,2).cuda(self.args.gpu)

        # training loop
        for i in range(self.args.num_iter):
            # there are three things we have to sample: f, x_s, x_i / x_b
            # The resample steps should be chosen s.t. step_x = a*step_xs = a*b*step_f
            # e.g. step_x = 10, step_xs = 100, step_f = 1000
            if i % self.args.resample_step_f == 0:
                # f_s   = self.prepare_f(x_s)
                # f_int = self.prepare_f(x_int)
                print ("Resampled f")
                self.prepare_f()

            if i % self.args.resample_step_xs == 0:
                print ("Resampled x_s")
                x_s = self.prepare_xs()
                f_s = self.f(x_s)

            if i % self.args.resample_step_x == 0:
                print ("Resampled x")
                x_int, x_bdry, _ = self.prepare_data()
                f_int            = self.f(x_int)
                # f_bdry           = self.f(x_bdry)
                
            # set this flag before forwarding it through the model
            x_int.requires_grad = True
            # get u(x;theta) on int and bdry
            u_int  = self.model(x_int, x_s, f_s)  # N_int  x 1
            u_bdry = self.model(x_bdry, x_s, f_s) # N_bdry x 1

            # compute grad_x(u): N_int x d
            dSdx = torch.autograd.grad(u_int, x_int, grad_outputs=torch.ones_like(u_int), \
                retain_graph=True,create_graph=True,only_inputs=True)[0]

            # loss on the interior
            L_int = 1/u_int.shape[0] * (1/2 * torch.norm(dSdx)**2 - torch.sum(u_int * f_int))

            # loss on the boundary
            L_bdry = 1/u_bdry.shape[0] * torch.norm(u_bdry)**2    

            # overall loss
            loss = L_int + self.args.lbd_1 * L_bdry
            loss.backward()
            self.optimizer.step()

            # logging training history
            if (i+1) % self.args.log_step == 0:
                print ("Loss value: {:.2f}, L_int:{:.2f}, L_bdry:{:.2f}".format(loss.item(), \
                    L_int.item(), L_bdry.item()))
                # print ("Model weight norm: {:.2f}".format(sum([torch.norm(p.data)**2 for p in self.model.parameters()] )))
                # # calculate error
                # self.model.eval()
                # with torch.no_grad():
                #     u_nn    = self.model(grid)      # N^2 x 1
                #     u_exact = self.exact_soln(grid) # N^2 x 1
                #     u_nn_grid = u_nn.view(N, N)
                #     u_exact   = u_exact.view(N, N)

                # diff  = torch.abs(u_nn_grid - u_exact)
                # # L_inf error
                # e_inf = torch.max(diff)
                # e_1   = torch.sum(diff)
                # # print("L_inf error: {:.4f}, Mean L_1 error: {:.4f}".format(float(e_inf),float(e_1)/N**2 ))
                self.model.set_u_net_param(x_s, f_s)
                L = - torch.stack([
                    torch.trace(torch.autograd.functional.hessian(self.model.u_net_fixed_param, x_i))
                    for x_i in torch.unbind(x_int, dim=0)
                    ], dim=0).reshape(-1,1)
                residual = 1/u_int.shape[0] * torch.sum(torch.abs(L - f_int))
                print("|-Delta u - f|: {:.4f}".format(residual.item()))
                print("LR: {}".format(str(self.optimizer.param_groups[0]['lr'])))
                

                # save data to txt
                self.plot_logger.log('iter_num', i+1)
                self.plot_logger.log('loss', float(loss))
                # self.plot_logger.log('e_inf', float(e_inf))

            # LR decay
            if self.args.adaptive_schedule:
                self.scheduler.step(loss)
            else:
                self.scheduler.step()

            # save model
            if (i + 1) % self.args.snapshot_interval == 0:
                    self.snapshot(i + 1)

        # save model
        self.snapshot(self.args.num_iter)




    

    # =================================================================================== #
    #                           U-Net for a fixed forcing f(x)                            #
    # =================================================================================== #

    def poisson_simple2(self):
        """Computes the solution to the 2D poisson equation:
        -delta(u) = pi^2/4 * [sin(pi/2*x) + sin(pi/2*y)] on the domain [0,1]^2
        with Dirichlet BC: u = [sin(pi/2*x) + sin(pi/2*y)] on the boundary.

        The exact solution is: u^* = [sin(pi/2*x) + sin(pi/2*y)]

        This time, we do not go through the variational objective.
        Instead, we simply try to satisfy the PDE itself.
        Obj: |-delta u - f|^2 + |u - g|^2

        Glossary:
            x_int (tensor): points sampled in the interior: N_int x d
            x_bdry (tensor): points sampled on the boundary:  N_bdry x d
            u_int (tensor): = network(x_int), values in the interior: N_int x 1
            u_bdry (tensor): = network(x_bdry), values on the boundary, N_bdry x 1
            loss_int (tensor): objective in the interior: scalar
            loss_bdry (tensor): objective on the boundary: scalar
            grid (tensor): N^2 x d regular grid
            N (int): size of the grid
            iter_num (int): iteration number
            loss (tensor): = g_int + g_bdry, final objective
        """
        self.init_model()

        # # construct 2d grid
        # grid_oneDim = torch.arange(0, 1+self.args.mesh_size, step=self.args.mesh_size)
        # grid        = torch.meshgrid(grid_oneDim, grid_oneDim)             # a tuple of N x N tensors
        # grid_2d     = torch.stack(grid, dim=-1)                            # N x N x 2
        # N           = grid_2d.shape[0]
        # grid        = grid_2d.view(-1,2).cuda(self.args.gpu)

        self.prepare_f()

        # training loop
        for i in range(self.args.num_iter):
            # sample points as data
            if i % self.args.resample_step_xs == 0:
                print ("Resampled x_s")
                x_s = self.prepare_xs()
                f_s = self.f(x_s)

            if i % self.args.resample_step_x == 0:
                print ("Resampled x")
                x_int, x_bdry, _ = self.prepare_data()
                f_int            = self.f(x_int)
                # f_bdry           = self.f(x_bdry)

            # send data to GPU
            if self.args.use_gpu:
                x_int  = x_int.cuda(self.args.gpu)
                x_bdry = x_bdry.cuda(self.args.gpu)
            
            self.model.train()
            self.model.zero_grad()

            # set this flag before forwarding it through the model
            x_int.requires_grad = True
            # get u(x;theta) on int and bdry
            u_int  = self.model(x_int)  # N_int x 1
            u_bdry = self.model(x_bdry) # N_bdry x 1

            # # compute -delta(u)
            # # This is what we want ---> delta_u = torch.trace(torch.autograd.functional.hessian(self.model, x_int)) <----
            # # currently, Pytorch does not support the batch version of hessian, i.e. can only calculate for a batch size of 1
            # # so we use a bootlegged np.apply_along_axis for it.
            # # create_graph=True because we need Pytorch to compute gradients wrt L
            # L = - torch.stack([
            #     torch.trace(torch.autograd.functional.hessian(self.model, x_i, create_graph=True)) 
            #     for x_i in torch.unbind(x_int, dim=0)
            #     ], dim=0).reshape(-1,1)
            
            # compute grad_x(u): N_int x d
            dudx = torch.autograd.grad(u_int, x_int, grad_outputs=torch.ones_like(u_int), \
                retain_graph=True,create_graph=True,only_inputs=True)[0]

            # zero out grads in model since these gradients are from u_int
            # They are not what the ones that we should update theta over
            self.model.zero_grad()

            # evaluate loss on the int and the bdry
            # loss_int  = 1 / 2 * torch.norm(L - np.pi**2/4 * self.forcing(x_int).view(-1,1))**2
            loss_int  = 1 / u_int.shape[0] * (1/2 * torch.norm(dudx)**2 - torch.sum(u_int * f_int))
            loss_bdry = 1 / u_bdry.shape[0] * torch.norm(u_bdry)**2
            
            # overall loss
            loss = loss_int + self.args.lbd_1 * loss_bdry
            loss.backward()

            self.optimizer.step()

            # logging training history
            if (i+1) % self.args.log_step == 0:
                print ("Loss value: {:.2f}, L_int:{:.2f}, L_bdry:{:.2f}".format(loss.item(), \
                    loss_int.item(), loss_bdry.item()))

                # self.model.set_u_net_param(x_s, f_s)
                L = - torch.stack([
                    torch.trace(torch.autograd.functional.hessian(self.model, x_i))
                    for x_i in torch.unbind(x_int, dim=0)
                    ], dim=0).reshape(-1,1)
                residual = 1/u_int.shape[0] * torch.sum(torch.abs(L - f_int))
                print("|-Delta u - f|: {:.4f}".format(residual.item()))
                print("LR: {}".format(str(self.optimizer.param_groups[0]['lr'])))
                
                # save data to txt
                self.plot_logger.log('iter_num', i+1)
                self.plot_logger.log('loss', float(loss))
                # self.plot_logger.log('e_inf', float(e_inf))

            # LR decay
            if self.args.adaptive_schedule:
                self.scheduler.step(loss)
            else:
                self.scheduler.step()

            # save model
            if (i + 1) % self.args.snapshot_interval == 0:
                    self.snapshot(i + 1)

        # save model
        self.snapshot(self.args.num_iter)


    def poisson_simple(self):
        """Computes the solution to the 2D poisson equation:
        -delta(u) = pi^2/4 * [sin(pi/2*x) + sin(pi/2*y)] on the domain [0,1]^2
        with Dirichlet BC: u = [sin(pi/2*x) + sin(pi/2*y)] on the boundary.

        The exact solution is: u^* = [sin(pi/2*x) + sin(pi/2*y)]

        Glossary:
            x_int (tensor): points sampled in the interior: N_int x d
            x_bdry (tensor): points sampled on the boundary:  N_bdry x d
            u_int (tensor): = network(x_int), values in the interior: N_int x 1
            u_bdry (tensor): = network(x_bdry), values on the boundary, N_bdry x 1
            g_int (tensor): objective in the interior: scalar
            g_bdry (tensor): objective on the boundary: scalar
            grid (tensor): N^2 x d regular grid
            N (int): size of the grid
            iter_num (int): iteration number
            loss (tensor): = g_int + g_bdry, final objective
        """
        self.init_model()

        # construct 2d grid
        grid_oneDim = torch.arange(0, 1+self.args.mesh_size, step=self.args.mesh_size)
        grid        = torch.meshgrid(grid_oneDim, grid_oneDim)             # a tuple of N x N tensors
        grid_2d     = torch.stack(grid, dim=-1)                            # N x N x 2
        N           = grid_2d.shape[0]
        grid        = grid_2d.view(-1,2).cuda(self.args.gpu)

        # training loop
        for i in range(self.args.num_iter):
            # sample points as data
            if i % self.args.resample_step_x == 0:
                x_int, x_bdry, _, _ = self.prepare_data()

            # send data to GPU
            if self.args.use_gpu:
                x_int  = x_int.cuda(self.args.gpu)
                x_bdry = x_bdry.cuda(self.args.gpu)
            
            self.model.train()
            self.model.zero_grad()

            # set this flag before forwarding it through the model
            x_int.requires_grad = True
            # get u(x;theta) on int and bdry
            u_int  = self.model(x_int)  # N_int x 1
            u_bdry = self.model(x_bdry) # N_bdry x 1

            # compute grad_x(u)
            dudx = torch.autograd.grad(u_int, x_int, grad_outputs=torch.ones_like(u_int), \
                retain_graph=True,create_graph=True,only_inputs=True)[0]

            # # # no need to retain grad on x_int anymore
            # x_int.grad          = None
            # x_int.requires_grad = False

            # zero out grads in model since these gradients are from u_int
            # They are not what the ones that we should update theta over
            self.model.zero_grad()

            # evaluate g(x,theta), vectorized 
            g_int = 1/u_int.shape[0] * (1/2 * torch.norm(dudx)**2 - np.pi**2/4* torch.sum(u_int * self.forcing(x_int).view(-1,1)) )
            g_bdry = self.args.lbd_1 / u_bdry.shape[0] * torch.norm(u_bdry - self.forcing(x_bdry).view(-1,1))**2
            
            # overall variational objective
            loss = g_int + g_bdry
            loss.backward()

            self.optimizer.step()

            # logging training history
            self.log(x_int, x_bdry, u_int, u_bdry, g_int, g_bdry, grid, N, i, loss)
            if (i+1) % self.args.log_step == 0:
                L = - torch.stack([
                    torch.trace(torch.autograd.functional.hessian(self.model, x_i))
                    for x_i in torch.unbind(x_int, dim=0)
                    ], dim=0).reshape(-1,1)
                residual = torch.max(torch.abs(L - self.forcing(x_int).view(-1,1)))
                print("|-Delta u - f|: {:.4f}".format(residual.item()))
                self.plot_logger.log('residual', float(residual))

            # LR decay
            if self.args.adaptive_schedule:
                self.scheduler.step(loss)
            else:
                self.scheduler.step()

            # save model
            if (i + 1) % self.args.snapshot_interval == 0:
                    self.snapshot(i + 1)

        # save model
        self.snapshot(self.args.num_iter)


    def U_Neumann(self):
        self.init_model()

        # construct 2d grid
        grid_1D     = torch.arange(0, 1+self.args.mesh_size, step=self.args.mesh_size)
        grid        = torch.meshgrid(grid_1D, grid_1D)             # a tuple of N x N tensors
        grid_2d     = torch.stack(grid, dim=-1)                            # N x N x 2
        N           = grid_2d.shape[0]
        grid        = grid_2d.view(-1,2).cuda(self.args.gpu)

        # sample 1 f and keep it fixed
        self.prepare_f()
        u_exact   = self.exact_soln(grid).view(N, N) # N x N

        # containers
        loss_sum  = 0
        e_inf_sum = 0
        e_1_sum   = 0
        e_rel_sum = 0

        # training loop
        for i in range(self.args.num_iter):
            # sample points as data
            if i % self.args.resample_step_x == 0:
                # print ("Resampled x")
                x_int, _, _, vol = self.prepare_data()
                f_int            = self.f(x_int)
            
            self.model.train()
            self.model.zero_grad()

            # set this flag before forwarding it through the model
            x_int.requires_grad = True
            # get u(x;theta) on int and bdry
            u_int  = self.model(x_int)  # N_int x 1
            # u_bdry = self.model(x_bdry) # N_bdry x 1

            # compute grad_x(u)
            dudx = torch.autograd.grad(u_int, x_int, grad_outputs=torch.ones_like(u_int), \
                retain_graph=True,create_graph=True,only_inputs=True)[0]

            # loss on the interior
            L_int = vol/u_int.shape[0] * (1/2*torch.norm(dudx)**2 - torch.sum(u_int * f_int)) + \
                    1/2*(vol/u_int.shape[0] * torch.sum(u_int))**2

            # No loss on the boundary

            # overall loss
            loss = L_int
            loss.backward()
            self.optimizer.step()

            # recording history
            loss_sum += float(loss)
            self.model.eval()
            with torch.no_grad():
                u_nn      = self.model(grid)      # N^2 x 1
                u_nn_grid = u_nn.view(N, N)
            # L_inf error
            diff  = torch.abs(u_nn_grid - u_exact)
            e_inf = torch.max(diff)
            # L_1 error
            e_1   = torch.mean(diff)
            # relative L_2 error
            e_rel = torch.norm(u_nn_grid - u_exact) / torch.norm(u_exact)
            # accumulate
            e_1_sum   += float(e_1)
            e_inf_sum += float(e_inf)
            e_rel_sum += float(e_rel)

            # logging training history
            if (i+1) % self.args.log_step == 0:

                # average error over an 'epoch'
                epoch     = (self.args.start_iter + i+1) // self.args.log_step
                e_1_avg   = float(e_1_sum) / self.args.log_step
                e_inf_avg = float(e_inf_sum) / self.args.log_step
                e_rel_avg = float(e_rel_sum) / self.args.log_step 
                loss_avg  = float(loss_sum) / self.args.log_step
                loss_thry = self.obj_min()

                print("Loss value: {:.5f}, Theoretical Min:{:.5f}".format(loss_avg, loss_thry.item()))
                print("L_inf error: {:.4f}, Mean Abs. error: {:.4f}, Relative L_2 error:{:.3f}".format(e_inf_avg, e_1_avg, e_rel_avg))
                print("Model weight norm: {:.2f}".format(sum([torch.norm(p.data)**2 for p in self.model.parameters()] )))
                # self.model.set_u_net_param(x_s, f_s)
                # L = - torch.stack([
                #     torch.trace(torch.autograd.functional.hessian(self.model.u_net_fixed_param, x_i))
                #     for x_i in torch.unbind(x_int, dim=0)
                #     ], dim=0).reshape(-1,1)
                # residual = torch.max(torch.abs(L - f_int))
                # print("|-Delta u - f|: {:.4f}".format(residual.item()))
                print("LR: {}".format(str(self.optimizer.param_groups[0]['lr'])))
                
                # save data to txt
                self.plot_logger.log('iter_num', self.args.start_iter + i+1)
                self.plot_logger.log('epoch', epoch)
                self.plot_logger.log('loss', loss_avg)
                self.plot_logger.log('loss_thry', float(loss_thry))
                self.plot_logger.log('e_inf', e_inf_avg)
                self.plot_logger.log('e_1', e_1_avg)
                self.plot_logger.log('e_rel', e_rel_avg)
                
                # reset history containers
                loss_sum  = 0
                e_inf_sum = 0
                e_1_sum   = 0
                e_rel_sum = 0

            # LR decay
            if self.args.adaptive_schedule:
                self.scheduler.step(loss)
            else:
                self.scheduler.step()

            # save model
            if (i + 1) % self.args.snapshot_interval == 0:
                    self.snapshot(self.args.start_iter + i + 1)

        # save model
        self.snapshot(self.args.start_iter + self.args.num_iter)

        # visualize after training
        self.model.eval()
        with torch.no_grad():
            u_nn    = self.model(grid)      # N^2 x 1
        u_nn_grid = u_nn.view(N, N)

        scipy.io.savemat(self.plot_dir + 'grid.mat',      dict(x=grid_1D.cpu().numpy(), y = grid_1D.cpu().numpy()))
        scipy.io.savemat(self.plot_dir + 'u_exact.mat',   dict(u_exact=u_exact.cpu().numpy()))
        scipy.io.savemat(self.plot_dir + 'u_nn.mat',      dict(u_nn=u_nn_grid.cpu().numpy()))
        scipy.io.savemat(self.plot_dir + 'e_abs.mat',     dict(e_abs=diff.cpu().numpy()))


    def solution_map_debug(self):
        self.init_model()

        # # construct 2d grid
        grid_oneDim = torch.arange(0, 1+self.args.mesh_size, step=self.args.mesh_size)
        grid        = torch.meshgrid(grid_oneDim, grid_oneDim)             # a tuple of N x N tensors
        grid_2d     = torch.stack(grid, dim=-1)                            # N x N x 2
        N           = grid_2d.shape[0]
        grid        = grid_2d.view(-1,2).cuda(self.args.gpu)

        # training loop
        for i in range(self.args.num_iter):
            # there are three things we have to sample: f, x_s, x_i / x_b
            # The resample steps should be chosen s.t. step_x = a*step_xs = a*b*step_f
            # e.g. step_x = 10, step_xs = 100, step_f = 1000
            if i % self.args.resample_step_f == 0:
                # f_s   = self.prepare_f(x_s)
                # f_int = self.prepare_f(x_int)
                print ("Resampled f")
                self.prepare_f()

            if i % self.args.resample_step_xs == 0:
                print ("Resampled x_s")
                x_s = self.prepare_xs()
                f_s = self.f(x_s)

            ##### method 1 #####
            param = torch.tensor([[ 0.0290, -0.1533,  0.2877]]).cuda(self.args.gpu)
            # param = torch.tensor([[ 0.0796, -0.1555,  0.2705]]).cuda(self.args.gpu)
            x     = torch.tensor([0.9457, 0.6955]).cuda(self.args.gpu)

            self.model.u_net_fixed_param.set_param(param)
            out = self.model.u_net_fixed_param(x)
            H = torch.autograd.functional.hessian(self.model.u_net_fixed_param, x)

            ##### method 2 #####
            self.model.set_u_net_param(x_s, f_s)
            H = torch.autograd.functional.hessian(self.model.u_net_fixed_param, x)

            # logging training history
            if (i+1) % self.args.log_step == 0:
                print ("Loss value: {:.2f}, L_int:{:.2f}".format(loss.item(), L_int.item()))
                # print ("Model weight norm: {:.2f}".format(sum([torch.norm(p.data)**2 for p in self.model.parameters()] )))
                # calculate error
                self.model.eval()
                with torch.no_grad():
                    u_nn      = self.model(grid, x_s, f_s)    # N^2 x 1
                    u_exact   = self.exact_soln(grid)         # N^2 x 1
                    u_nn_grid = u_nn.view(N, N)
                    u_exact   = u_exact.view(N, N)

                diff  = torch.abs(u_nn_grid - u_exact)
                # L_inf error
                e_inf = torch.max(diff)
                e_1   = torch.sum(diff)
                print("L_inf error: {:.4f}, Mean L_1 error: {:.4f}".format(float(e_inf),float(e_1)/N**2 ))
                self.model.set_u_net_param(x_s, f_s)
                L = - torch.stack([
                    torch.trace(torch.autograd.functional.hessian(self.model.u_net_fixed_param, x_i))
                    for x_i in torch.unbind(x_int, dim=0)
                    ], dim=0).reshape(-1,1)
                residual = torch.max(torch.abs(L - f_int))
                print("|-Delta u - f|: {:.4f}".format(residual.item()))
                print("LR: {}".format(str(self.optimizer.param_groups[0]['lr'])))
                
                # save data to txt
                self.plot_logger.log('iter_num', i+1)
                self.plot_logger.log('loss', float(loss))
                self.plot_logger.log('e_inf', float(e_inf))

            # LR decay
            if self.args.adaptive_schedule:
                self.scheduler.step(loss)
            else:
                self.scheduler.step()

            # save model
            if (i + 1) % self.args.snapshot_interval == 0:
                    self.snapshot(i + 1)

        # save model
        self.snapshot(self.args.num_iter)


    def log(self, x_int, x_bdry, u_int, u_bdry, g_int, g_bdry, grid, N, iter_num, loss):
        """prints and saves the relevant training history

        Args:
            x_int (tensor): points sampled in the interior: N_int x d
            x_bdry (tensor): points sampled on the boundary:  N_bdry x d
            u_int (tensor): = network(x_int), values in the interior: N_int x 1
            u_bdry (tensor): = network(x_bdry), values on the boundary, N_bdry x 1
            g_int (tensor): objective in the interior: scalar
            g_bdry (tensor): objective on the boundary: scalar
            grid (tensor): N^2 x d regular grid
            N (int): size of the grid
            iter_num (int): iteration number
            loss (tensor): = g_int + g_bdry, final objective
        """

        if (iter_num+1) % self.args.log_step == 0:
            print ("Objective value: {:.2f}, g_int:{:.2f}, g_bdry:{:.2f}".format(loss.item(), \
                g_int.item(), g_bdry.item()))
            u_int_exact  = self.exact_soln(x_int).view(-1,1)
            dudx_exact   = self.dudx_exact(x_int)
            g_int_exact = 1/u_int.shape[0] * (1/2 * torch.norm(dudx_exact)**2 \
                - np.pi**2/4* torch.sum(u_int_exact * self.forcing(x_int).view(-1,1)) )
            print ("Theoretical obj value: {:.2f}".format(float(g_int_exact)))
            print ("Model weight norm: {:.2f}".format(sum([torch.norm(p.data)**2 for p in self.model.parameters()] )))
            # calculate error
            self.model.eval()
            with torch.no_grad():
                u_nn    = self.model(grid)      # N^2 x 1
                u_exact = self.exact_soln(grid) # N^2 x 1
                u_nn_grid = u_nn.view(N, N)
                u_exact   = u_exact.view(N, N)

            diff  = torch.abs(u_nn_grid - u_exact)
            # diff = torch.abs(self.exact_soln(x_int).view(-1,1) - u_int)
            # L_inf error
            e_inf = torch.max(diff)
            e_1   = torch.sum(diff)
            print("LR: {}".format(str(self.optimizer.param_groups[0]['lr'])))
            print("L_inf error: {:.4f}, Mean L_1 error: {:.4f}".format(float(e_inf),float(e_1)/N**2 ))

            # save data to txt
            self.plot_logger.log('iter_num', iter_num+1)
            self.plot_logger.log('loss', float(loss))
            self.plot_logger.log('e_inf', float(e_inf))


    def visualize_UPS(self):
        if self.args.model_path == '':
            raise RuntimeError("Need a pretrained model to compare with the exact solution")
        self.init_model()
        # self.load_pretrain(path=self.args.model_path)
        self.model.eval()

        # # construct 2d grid
        grid_oneDim = torch.arange(0, 1+1e-10, step=self.args.mesh_size)
        grid        = torch.meshgrid(grid_oneDim, grid_oneDim)             # a tuple of N x N tensors
        grid_2d     = torch.stack(grid, dim=-1)                            # N x N x 2
        N           = grid_2d.shape[0]
        grid        = grid_2d.view(-1,2).cuda(self.args.gpu)
        if self.args.batch_f:
            grid = torch.repeat_interleave(grid.unsqueeze(0), self.args.N_f, dim=0) # N_f x N^d x 1

        # sample a new batch of functions
        self.prepare_f()
        x_s = self.prepare_xs()    # N_f x N_s x d
        f_s = self.f(x_s)          # N_f x N_s x 1

        # compute solutions
        with torch.no_grad():
            u_nn      = self.model(grid, x_s, f_s)    # N_f x N^d x 1
            u_exact   = self.exact_soln(grid)         # N_f x N^d x 1
        
        # compute error
        diff  = torch.abs(u_nn - u_exact)
        e_inf      = torch.mean(torch.max(diff.squeeze(-1), dim=-1)[0])
        e_1_mean   = torch.mean(diff)
        e_rel      = torch.mean(torch.norm(u_nn.squeeze(-1)-u_exact.squeeze(-1), dim=-1) / torch.norm(u_exact.squeeze(-1), dim=-1))

        print("L_inf error: {:.4f}, Mean L_1 error: {:.4f}, Relative L_2 error:{:.3f}".format(e_inf, e_1_mean, e_rel))

        # save plotting data
        for i in range(self.args.N_f):
            scipy.io.savemat(self.plot_dir + 'grid_{}.mat'.format(i),      dict(x=grid_oneDim.cpu().numpy(), y=grid_oneDim.cpu().numpy()))
            scipy.io.savemat(self.plot_dir + 'u_exact_{}.mat'.format(i),   dict(u_exact=u_exact[i].view([N]*self.args.input_dim).cpu().numpy()))
            scipy.io.savemat(self.plot_dir + 'u_nn_{}.mat'.format(i),      dict(u_nn=u_nn[i].view([N]*self.args.input_dim).cpu().numpy()))
            scipy.io.savemat(self.plot_dir + 'e_abs_{}.mat'.format(i),     dict(e_abs=diff[i].view([N]*self.args.input_dim).cpu().numpy()))



    def visualize_simple_possion(self):
        """Visualize and compare the computed solution to the 2D poisson equation to the exact values.

        Raises:
            RuntimeError: Need a pretrained model to run this method.
        """
        if self.args.model_path == '':
            raise RuntimeError("Need a pretrained model to compare with the exact solution")
        self.init_model()
        # self.load_pretrain(path=self.args.model_path)

        # construct 2d grid
        grid_1D = torch.arange(0,1+self.args.mesh_size, step=self.args.mesh_size)
        grid    = torch.meshgrid(grid_1D, grid_1D)                     # a tuple of N x N tensors
        grid_2D = torch.stack(grid, dim=-1)                            # N x N x 2
        N       = grid_2D.shape[0]
        grid    = grid_2D.view(-1,2).cuda(self.args.gpu)
        # may need to change later if grid gets too big
        self.model.eval()
        with torch.no_grad():
            u_nn    = self.model(grid)      # N^2 x 1
        u_exact = self.exact_soln(grid)     # N^2 x 1
        u_nn_grid = u_nn.view(N, N)
        u_exact   = u_exact.view(N, N)

        diff  = torch.abs(u_nn_grid - u_exact)
        # L_inf error
        e_inf = torch.max(diff)
        print("L_inf error: {:.4f}".format(e_inf.item()))

        scipy.io.savemat(self.plot_dir + 'grid.mat',      dict(x=grid_1D.cpu().numpy(), y = grid_1D.cpu().numpy()))
        scipy.io.savemat(self.plot_dir + 'u_exact.mat',   dict(u_exact=u_exact.cpu().numpy()))
        scipy.io.savemat(self.plot_dir + 'u_nn.mat',      dict(u_nn=u_nn_grid.cpu().numpy()))
        scipy.io.savemat(self.plot_dir + 'e_abs.mat', dict(e_abs=diff.cpu().numpy()))


    def visualize(self):
        # input: pretrained model (path), grid points to evaluate at, true solution
        # output: error plot
        raise NotImplementedError()


    def snapshot(self, epoch, mode=''):
        if mode != 'MFP':
            save_dir = os.path.join(self.save_dir, 'DRM')
            torch.save(self.model.state_dict(), save_dir + "_" + str(epoch) + '.pkl')
            print(f"Save model to {save_dir}_{str(epoch)}.pkl")
        else:
            save_dir = os.path.join(self.save_dir, 'rho')
            torch.save(self.rho.state_dict(), save_dir + "_" + str(epoch) + '.pkl')
            print(f"Save model to {save_dir}_{str(epoch)}.pkl")
            save_dir = os.path.join(self.save_dir, 'm')
            torch.save(self.m.state_dict(), save_dir + "_" + str(epoch) + '.pkl')
            print(f"Save model to {save_dir}_{str(epoch)}.pkl")

    def load_pretrain(self, path='', rho_path='', m_path='', mode=''):
        if mode != 'MFP':
            if path == '':
                raise RuntimeError("Pretrained model path cannot be empty.")
            state_dict = torch.load(path, map_location='cpu')
            self.model.load_state_dict(state_dict, strict=False)
            print(f"Load model from {path}")
        else:
            if rho_path == '':
                raise RuntimeError("Pretrained rho model path cannot be empty.")
            if m_path == '':
                raise RuntimeError("Pretrained m model path cannot be empty.")
            state_dict_rho = torch.load(rho_path, map_location='cpu')
            self.rho.load_state_dict(state_dict_rho, strict=False)
            print(f"Load rho model from {rho_path}")

            state_dict_m = torch.load(m_path, map_location='cpu')
            self.m.load_state_dict(state_dict_m, strict=False)
            print(f"Load model from {m_path}")