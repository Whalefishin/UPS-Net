% I/O
run_dir      = '../snapshot/';
exp_dir     = 'UPS_big_r_f=10_r_x=5_long';
exp_name    = '2D Poisson Neumann BC SP BigUNet: ';
save_folder  = strcat(run_dir, exp_dir, '/plots/');
run_folder   = strcat(run_dir, exp_dir, '/plot_data/');
% save name
session_name = strcat('_', exp_dir);
% create save folder if it doesn't exist
if ~exist(save_folder, 'dir')
    mkdir(save_folder)
end

% load data
N_f = 32;
for i=0:N_f-1
    load(strcat(run_folder,'e_abs_', num2str(i), '.mat'));
    load(strcat(run_folder,'u_exact_', num2str(i),'.mat'));
    load(strcat(run_folder,'u_nn_', num2str(i),'.mat'));
    load(strcat(run_folder,'grid_', num2str(i),'.mat'));

    rainbowMap = getRainbow();
    figure(1); colormap(rainbowMap);
    surf(x,y,u_exact); colorbar;
    xlabel('x'); ylabel('y');
    title(strcat(exp_name, ' exact solution'));
    saveas(gcf,strcat(save_folder, 'u_exact_', num2str(i) ,session_name,'.fig'))
    saveas(gcf,strcat(save_folder, 'u_exact_', num2str(i) ,session_name,'.png'))


    figure(2); colormap(rainbowMap);
    surf(x,y,u_nn); colorbar;
    xlabel('x'); ylabel('y');
    title(strcat(exp_name, ' solution computed by the NN'));
    saveas(gcf,strcat(save_folder, 'u_nn_', num2str(i) ,session_name,'.fig'))
    saveas(gcf,strcat(save_folder, 'u_nn_', num2str(i) ,session_name,'.png'))


    figure(3); colormap(rainbowMap);
    surf(x,y,e_abs); colorbar;
    xlabel('x'); ylabel('y');
    title(strcat(exp_name, ' absolute error'));
    saveas(gcf,strcat(save_folder, 'e_abs_', num2str(i),session_name,'.fig'))
    saveas(gcf,strcat(save_folder, 'e_abs_', num2str(i),session_name,'.png'))

end
