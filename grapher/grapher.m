% I/O
run_dir      = '../snapshot/';
exp_dir     = 'UNet_sp_slowerLR_reRun';
exp_name    = '2D Poisson Neumann BC Slower LR decay ';
save_folder  = strcat(run_dir, exp_dir, '/plots/');
run_folder   = strcat(run_dir, exp_dir, '/plot_data/');
% save name
session_name = strcat('_', exp_dir);
% create save folder if it doesn't exist
if ~exist(save_folder, 'dir')
    mkdir(save_folder)
end

% params
init_steps = 1; % start plotting at the x step

% load data
e_inf     = load(strcat(run_folder,'e_inf.txt'));
e_1       = load(strcat(run_folder,'e_1.txt'));
e_rel     = load(strcat(run_folder,'e_rel.txt'));
loss      = load(strcat(run_folder,'loss.txt'));
loss_thry = load(strcat(run_folder,'loss_thry.txt'));
iter_num  = load(strcat(run_folder,'iter_num.txt'));
% residual  = load(strcat(run_folder,'residual.txt'));
e_inf     = e_inf(init_steps:end);
e_1       = e_1(init_steps:end);
e_rel     = e_rel(init_steps:end);
loss      = loss(init_steps:end);
loss_thry = loss_thry(init_steps:end);
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
plot(iter_num, e_inf);
legend('Error');
xlabel('Iteration Number'); ylabel('Value'); grid on;
title({exp_name; 'Mean L_\infty Error vs. Iteration Number'});
set(gca, 'YScale', 'log')
saveas(gcf,strcat(save_folder, 'e_inf',session_name,'.fig'))
saveas(gcf,strcat(save_folder, 'e_inf',session_name,'.png'))

figure
plot(iter_num, e_rel);
legend('Error');
xlabel('Iteration Number'); ylabel('Value'); grid on;
title({exp_name; ' Relative L_2 Error vs. Iteration Number'});
set(gca, 'YScale', 'log')
saveas(gcf,strcat(save_folder, 'e_rel',session_name,'.fig'))
saveas(gcf,strcat(save_folder, 'e_rel',session_name,'.png'))


figure
plot(iter_num, e_1);
legend('Error');
xlabel('Iteration Number'); ylabel('Value'); grid on;
title({exp_name; ' Mean Absolute Error vs. Iteration Number'});
set(gca, 'YScale', 'log')
saveas(gcf,strcat(save_folder, 'e_1',session_name,'.fig'))
saveas(gcf,strcat(save_folder, 'e_1',session_name,'.png'))

figure
plot(iter_num, abs(loss - loss_thry));
legend('Error');
xlabel('Iteration Number'); ylabel('Value'); grid on;
title({exp_name; ' Loss gap vs. Iteration Number'});
saveas(gcf,strcat(save_folder, 'loss_gap',session_name,'.fig'))
saveas(gcf,strcat(save_folder, 'loss_gap',session_name,'.png'))

% figure
% plot(iter_num, residual);
% legend('|Delta u - f|');
% xlabel('Iteration Number'); ylabel('Value'); grid on;
% title(strcat(exp_name, ' Residual vs. Iteration Number'));
% saveas(gcf,strcat(save_folder, 'residual',session_name,'.fig'))
% saveas(gcf,strcat(save_folder, 'residual',session_name,'.png'))