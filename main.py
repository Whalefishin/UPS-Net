import argparse
import torch
import numpy as np
from solver import Solver



def parse_args():
    parser = argparse.ArgumentParser(description='DRM')
    # =================================================================================== #
    #                               Logistics & Misc                                      #
    # =================================================================================== #
    parser.add_argument('--run_choice',         type=str, default='poisson_simple', metavar='N',choices=[\
                                                'poisson_simple', 'U_Neumann', 'visualize', 'poisson_simple2', \
                                                'poisson_solution_map', 'UPS_Neumann', 'solution_map_debug',
                                                'visualize_UPS', 'MFP_naive', 'MFP_visualization', 'make_movie'])
    parser.add_argument('--mesh_size',          type=float, default=1e-1, help='Grid resolution for plotting the trained solution')
    parser.add_argument('--seed',               type=int, default=42,help='RNG seed')
    parser.add_argument('--snapshot_interval',  type=int, default=50000, metavar='N',help='Save snapshot every x epochs')
    parser.add_argument('--num_workers',        type=int, default=1, metavar='N',help='Number of worker processes used for data loading')
    parser.add_argument('--model_path',         type=str, default='', metavar='N',help='Path to load pretrained model')
    parser.add_argument('--rho_path',           type=str, default='', metavar='N',help='Path to load pretrained model')
    parser.add_argument('--m_path',             type=str, default='', metavar='N',help='Path to load pretrained model')
    parser.add_argument('--exp_name',           type=str, default="test_run", metavar='N',help='Name of the experiment')
    parser.add_argument('--overwrite_folder',   type=lambda x: (str(x).lower() == 'true'), default=False, help='Overwrite save folder')
    parser.add_argument('--log_step',           type=int, default=1000, help="Prints molecular scores on validation set every x batches")
    parser.add_argument('--use_gpu',            type=lambda x: (str(x).lower() == 'true'), default=True, help='Use GPU or not')
    parser.add_argument('--gpu',                type=int, default=0,help='id of gpu to use')
    parser.add_argument('--debug',              action='store_true', default=False, help='Debug mode')

    # =================================================================================== #
    #                               Training & Optimization                               #
    # =================================================================================== #
    parser.add_argument('--optimizer',          type=str, default='adam', metavar='N',choices=['adam', 'sgd'])
    parser.add_argument('--lr',                 type=float, default=1e-3,help='learning rate')
    parser.add_argument('--start_iter',         type=int, default=0,help='number of iterations to train')
    parser.add_argument('--num_iter',           type=int, default=50000,help='number of iterations to train')
    parser.add_argument('--weight_decay',       type=float, default=0,help='L2 regularization strength')
    parser.add_argument('--adaptive_schedule',  type=lambda x: (str(x).lower() == 'true'), default=False, help='True = ReduceLROnPlateau, False = StepLR')
    parser.add_argument('--lr_decay_gamma',     type=float, default=0.99,help='learning rate decay coefficient')
    parser.add_argument('--step_size',          type=int, default=2000, metavar='N',help='Scheduler step size ')

    # =================================================================================== #
    #                                Model & Architecture                                 #
    # =================================================================================== #
    parser.add_argument('--num_layers',         type=int, default=2, metavar='N',help='Dimension of the hidden layers in the Ritz net')
    parser.add_argument('--d',                  type=int, default=2, metavar='N',help='Dimension of x')
    parser.add_argument('--input_dim',          type=int, default=3, metavar='N',help='Dimension of the input (d+1)')
    parser.add_argument('--hidden_dim',         type=int, default=8, metavar='N',help='Dimension of the hidden layers in the P net')
    parser.add_argument('--hidden_list_u',      type=str, default='8,8,8', metavar='N',help='Dimension of the hidden layers in the U-net')
    parser.add_argument('--point_p_net',        type=lambda x: (str(x).lower() == 'true'), default=False, help='Whether to use point-net like modules in P-net')
    parser.add_argument('--pooling',            type=str, default='max', metavar='N',choices=['max', 'mean'], help="Pooling for P-Net")
    parser.add_argument('--act_p',              type=str, default='relu', metavar='N',choices=['relu', 'tanh'], help="Activation for the P-Net ")
    parser.add_argument('--act_u',              type=str, default='softplus', metavar='N',choices=['softplus', 'tanh'], help="Activation for the P-Net ")
    parser.add_argument('--tau_sp',             type=float, default=0.,help='value of tau in softplus')

    # =================================================================================== #
    #                                Data & Sampling                                      #
    # =================================================================================== #
    parser.add_argument('--N_int',              type=int, default=1024, metavar='N',help='Number of points to sample in the interior')
    parser.add_argument('--N_bdry',             type=int, default=1024, metavar='N',help='Number of points to sample on the boundary')
    parser.add_argument('--N_s',                type=int, default=1024, metavar='N',help='Number of points to sample for f')
    parser.add_argument('--N_f',                type=int, default=32, metavar='N',help='Number of points to sample for f')
    parser.add_argument('--resample_step_f',    type=int, default=1, help="Resample the forcing function f every x steps")
    parser.add_argument('--resample_step_xs',   type=int, default=1, help="Resample the locations of f to be evaluated on every x steps")
    parser.add_argument('--resample_step_x',    type=int, default=1, help="Resample x every n steps")
    parser.add_argument('--batch_f',            type=lambda x: (str(x).lower() == 'true'), default=True, help='Whether to forward a batch of f each time')
    parser.add_argument('--same_x',             type=lambda x: (str(x).lower() == 'true'), default=False, help='Whether to use the same points for x_s and x')
    parser.add_argument('--same_batch_xs',      type=lambda x: (str(x).lower() == 'true'), default=True, help='Whether to use the same points for x_s each batch')
    parser.add_argument('--xs_sampling',        type=str, default='grid', metavar='N',choices=['grid', 'unif', 'unif_grid'], help="Sampling technique for x_s")
    parser.add_argument('--num_grid_points_xs', type=int, default=30, help="If use regular grid for xs, the number of points on each dim.")
    parser.add_argument('--M',                  type=int, default=10, help="Max number to sum up to in the multivariate cosine basis")
    parser.add_argument('--b_mult',             type=float, default=1, help="Factor to multiply b_ij by. e.g. if use 2, then b_ij ~ U[-2,2]")

    # =================================================================================== #
    #                                       MFP                                           #
    # =================================================================================== #
    parser.add_argument('--MFP_sampling',       type=str, default='sqr', metavar='N',choices=['sqr', 'nonsqr'], help="Sampling technique for MFP")
    parser.add_argument('--MFP_ICTC',           type=str, default='sqr_circ', metavar='N',choices=['sqr_circ', 'gaussian'], help="Sampling technique for MFP")

    
    # =================================================================================== #
    #                             Penalty & Warm-up                                       #
    # =================================================================================== #
    parser.add_argument('--lbd_1',              type=float, default=500, help='(Eventual) Penalty weight on the BC/IC term')
    parser.add_argument('--lbd_2',              type=float, default=500, help='(Eventual) Penalty weight on the conservation term')
    parser.add_argument('--penalty_warmup',     type=str, default='none',choices=['none', 'ICBC', 'conserv', 'all'], help="Warmup strategy for lbd")
    parser.add_argument('--lbd_init',           type=float, default=10, help='penalty weight on the BC term')
    parser.add_argument('--p',                  type=float, default=0.5, help='penalty weight on the BC term')




    args = parser.parse_args()
    return args



def main():
    # get args
    args = parse_args()
    # seeding
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # run
    solver = Solver(args)
    solver.run()

    print ('Done!')




if __name__ == '__main__':
    main()