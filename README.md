# UPS Net: A Deep Learning Approach for solving PDEs

The UPS net is a deep learning architecture for approximating solutions for the Poisson equation. The model is based on the Deep Ritz method: https://arxiv.org/abs/1710.00211. This repository features:

1. An efficient implementation of the Deep Ritz method in Pytorch, where the gradients in the Ritz energy functional is computed via the AutoGrad engine. The sampling complexity grows slowly with respect to dimensionality, making it feasible to accurately approximate solutions to Poisson equations in extremely high dimensions.
2. An implementation for the UPS net that learns the solution map for the Poisson equation, which allows it to predict the solution for an arbitrary forcing term without re-training. This task is difficult in high dimensions due to challenges in sampling the forcing term.
3. A naive least square approach to solve variational mean-field games (MFP).



## Example Results:


### Learned solution map for the 2D Poisson equation


<!-- <p float="left">
  <img src="/results/SolutionMap/plots/u_exact_0_UPS_big_r_f=10_r_x=5_long.png" width="400" />
  <img src="/results/SolutionMap/plots/u_exact_0_UPS_big_r_f=10_r_x=5_long.png" width="400" /> 
  <img src="/results/SolutionMap/plots/u_exact_0_UPS_big_r_f=10_r_x=5_long.png" width="400" />
</p> -->

<p float="left">
  <img src="/results/SolutionMap/plots/u_exact_0_UPS_big_r_f=10_r_x=5_long.png" />
  <img src="/results/SolutionMap/plots/u_exact_0_UPS_big_r_f=10_r_x=5_long.png" /> 
</p>

### Optimal transport between two Gaussians



## Training:
<!-- 
Some general notes:

1. The choice of activation functions matters, try different options. 
2. LR decay schedule matters.
3. Network architecture matters - extremely small nets are surprisingly effective, and scaling (both in width and depth)
    may not give better results. -->


### 2D Poisson Equation, Dirlichlet BC:

```
python main.py --lr 1e-2 --log_step 50 --num_iter 50000 --overwrite_folder false --exp_name test_1 --hidden_dim 8 --lbd_1 500 --step_size 2000 --adaptive_schedule false --lr_decay_gamma 0.8 --N_int 1024 --N_bdry 1024 --batch_f false
```

Afterwards, the relevant training history can be found in 'snapshot/exp_name'.


### 2D Poisson Equation, Neumann BC:
```
python main.py --exp_name Neumann_debug --num_iter 50000 --run_choice poisson_simple_Neumann --batch_f false --N_int 1024 --step_size 2000 --lr 1e-3
```


### 2D Poisson Equation, Neumann BC, Solution Map:
```
python main.py --exp_name SNet --N_f 32 --log_step 1000 --num_iter 100000 --step_size 4000 --run_choice poisson_solution_map_Neumann
```


## Visualization

### 2D Poisson Equation, Neumann BC, Solution Map:
```
python main.py --exp_name visualize_SNet --model_path snapshot/S_net_N_f=32_fastResample_bigPNet_complete/models/DRM_300000.pkl --run_choice visualize_SNet --hidden_dim 128 --num_layers 3
```


### MFP, Gaussian IC & TC:
```
python main.py --run_choice MFP_visualization --rho_path snapshot/MFP_naive_nonsqr_lbd=50/models/rho_200000.pkl --m_path snapshot/MFP_naive_nonsqr_lbd=50/models/m_200000.pkl --overwrite_folder true --mesh_size 0.001
```
