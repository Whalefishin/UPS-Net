% I/O
run_dir      = '../snapshot/';
exp_dir     = 'MFP_naive_nonsqr_gaussian_thinner';
exp_name    = 'MFP naive GaussianICTC 2D Small U-Net LR=1e-3';
save_folder  = strcat(run_dir, exp_dir, '/plots/');
run_folder   = strcat(run_dir, exp_dir, '/plot_data/');
% save name
session_name = strcat('_', exp_dir);
% create save folder if it doesn't exist
if ~exist(save_folder, 'dir')
    mkdir(save_folder)
end

% params
init_steps = 5; % start plotting at the x step

% load data
l_conserv = load(strcat(run_folder,'l_conserv.txt'));
l_cost    = load(strcat(run_folder,'l_cost.txt'));
l_IC      = load(strcat(run_folder,'l_IC.txt'));
l_TC      = load(strcat(run_folder,'l_TC.txt'));
loss      = load(strcat(run_folder,'loss.txt'));
% loss_thry = load(strcat(run_folder,'loss_thry.txt'));
iter_num  = load(strcat(run_folder,'iter_num.txt'));
% residual  = load(strcat(run_folder,'residual.txt'));
l_conserv = l_conserv(init_steps:end);
l_cost    = l_cost(init_steps:end);
l_IC      = l_IC(init_steps:end);
l_TC      = l_TC(init_steps:end);
loss      = loss(init_steps:end);
% loss_thry = loss_thry(init_steps:end);
iter_num  = iter_num(init_steps:end);

%Plotting
figure
plot(iter_num, loss);
legend('Loss');
xlabel('Iteration Number'); ylabel('Value'); grid on;
title({exp_name; ' Loss vs. Iteration Number '});
saveas(gcf,strcat(save_folder, 'loss',session_name,'.fig'))
saveas(gcf,strcat(save_folder, 'loss',session_name,'.png'))

figure
plot(iter_num, l_conserv);
legend('L_{conserv}');
xlabel('Iteration Number'); ylabel('Value'); grid on;
title({exp_name; 'Conservation Loss vs. Iteration Number'});
set(gca, 'YScale', 'log')
saveas(gcf,strcat(save_folder, 'l_conserv',session_name,'.fig'))
saveas(gcf,strcat(save_folder, 'l_conserv',session_name,'.png'))

figure
plot(iter_num, l_IC);
legend('L_{IC}');
xlabel('Iteration Number'); ylabel('Value'); grid on;
title({exp_name; 'IC Loss vs. Iteration Number'});
set(gca, 'YScale', 'log')
saveas(gcf,strcat(save_folder, 'l_IC',session_name,'.fig'))
saveas(gcf,strcat(save_folder, 'l_IC',session_name,'.png'))

figure
plot(iter_num, l_TC);
legend('L_{TC}');
xlabel('Iteration Number'); ylabel('Value'); grid on;
title({exp_name; 'TC Loss vs. Iteration Number'});
set(gca, 'YScale', 'log')
saveas(gcf,strcat(save_folder, 'l_TC',session_name,'.fig'))
saveas(gcf,strcat(save_folder, 'l_TC',session_name,'.png'))

figure
plot(iter_num, l_cost);
legend('L_{cost}');
xlabel('Iteration Number'); ylabel('Value'); grid on;
title({exp_name; 'Cost Loss vs. Iteration Number'});
% set(gca, 'YScale', 'log')
saveas(gcf,strcat(save_folder, 'l_cost',session_name,'.fig'))
saveas(gcf,strcat(save_folder, 'l_cost',session_name,'.png'))

% figure
% plot(iter_num, abs(loss - loss_thry));
% legend('Error');
% xlabel('Iteration Number'); ylabel('Value'); grid on;
% title({exp_name; ' Loss gap vs. Iteration Number'});
% saveas(gcf,strcat(save_folder, 'loss_gap',session_name,'.fig'))
% saveas(gcf,strcat(save_folder, 'loss_gap',session_name,'.png'))

% figure
% plot(iter_num, residual);
% legend('|Delta u - f|');
% xlabel('Iteration Number'); ylabel('Value'); grid on;
% title(strcat(exp_name, ' Residual vs. Iteration Number'));
% saveas(gcf,strcat(save_folder, 'residual',session_name,'.fig'))
% saveas(gcf,strcat(save_folder, 'residual',session_name,'.png'))