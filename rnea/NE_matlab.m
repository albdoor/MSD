% Clear previous variables and close any open figures
clear all; close all; clc;

% Current working directory and file listing
% current_dir = pwd();
% disp('Current Working Directory:');
% disp(current_dir);
% disp('Files in Current Directory:');
% dir_contents = dir();
% {dir_contents.name}

% Define the manipulator links
links = {
    {0, 0, 1.0, 1.0, diag([0.0, 0.0, 0.1]), 1, 0.0}  % Link 1
};

% Time steps
time = linspace(0, 10, 1000);
torques = zeros(length(time), 1);
torquesLE = zeros(length(time), 2);

% Load joint data
[q_csv, qd_csv, qdd_csv] = load_joint_data('simplePendLE.mat');

% Print data shapes and types
disp('Shape of q:'); disp(size(q_csv));
disp('Shape of qd:'); disp(size(qd_csv));
disp('Shape of qdd:'); disp(size(qdd_csv));

% Gravity
g = 9.81;
gravity = [0; -g; 0];

% Compute torques
for t_idx = 1:length(time)
    q = q_csv(t_idx);     % Joint positions
    qd = qd_csv(t_idx);   % Joint velocities
    qdd = qdd_csv(t_idx); % Joint accelerations
    
    % Compute torques using RNEA
    torques(t_idx) = rnea(q, qd, qdd, links, gravity);
    
    % Compute input torques
    torquesLE(t_idx, :) = tau_input(time(t_idx));
end

% Plotting
figure('Position', [100, 100, 1200, 500]);
subplot(1, 2, 1);
plot(time, torques, 'b-', 'LineWidth', 2, 'DisplayName', 'Torque 1');
hold on;
plot(time, torquesLE(:, 1), 'r--', 'LineWidth', 2, 'DisplayName', 'Torque 1 Input');
plot(time, qd_csv, 'g-.', 'LineWidth', 2, 'DisplayName', 'qd input');
xlabel('Time (s)');
ylabel('Torque (Nm)');
title('Joint Torques Over Time');
legend('show');
grid on;

% Display final figure
subplot(1, 2, 2);
plot(time, torques, 'b-', 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Torque (Nm)');
title('RNEA Computed Torques');
grid on;

% All function definitions go at the end of the file

% Tau input function
function tau = tau_input(t)
    tau = [2 * sin(0.5 * t); 1.5 * cos(0.5 * t)];
end

% Load joint data function
function [q, qd, qdd] = load_joint_data(mat_filename)
    % Load data from .mat file
    loaded_data = load(mat_filename);

    % Extract values from the 'sol' and 'theta_ddot' variables
    q = loaded_data.sol(:, 1);    % First column of sol is q
    qd = loaded_data.sol(:, 2);   % Second column of sol is qd
    qdd = loaded_data.theta_ddot; % theta_ddot is directly used as qdd
    
    q = q(:);
    qd = qd(:);
    qdd = qdd(:);
end

% Rotation matrix to base
function R_direct = rotation_matrix_to_base(q, links, link_idx)
    % Start with cumulative angle
    cumul_angle = 0;
    
    % Accumulate rotations up to the specified link
    for i = 1:(link_idx + 1)
        theta = links{i}(1);
        j_type = links{i}(6);
        
        % Only revolute joints contribute to rotation
        if j_type == 1
            cumul_angle = cumul_angle + theta;
        end
    end
    
    % Direct rotation matrix calculation
    R_direct = [
        cos(cumul_angle), -sin(cumul_angle), 0;
        sin(cumul_angle), cos(cumul_angle), 0;
        0, 0, 1
    ];
end

% Rotation matrix function
function R = rotation_matrix(theta, alpha)
    R = [
        cos(theta), -cos(alpha) * sin(theta), sin(alpha) * sin(theta);
        sin(theta), cos(alpha) * cos(theta), -sin(alpha) * cos(theta);
        0, sin(alpha), cos(alpha)
    ];
end

% S vector function
function s = s_vector(theta, l)
    if length(theta) == 1
        s = [-l/2 * cos(theta(1)); 
             -l/2 * sin(theta(1)); 
             0];
    else
        theta_fin = sum(theta);
        s = [-l/2 * cos(theta_fin); 
             -l/2 * sin(theta_fin); 
             0];
    end
end

% P star vector function
function p = p_star_vector(theta, l)
    if length(theta) == 1
        p = [l * cos(theta(1)); 
             l * sin(theta(1)); 
             0];
    else
        theta_fin = sum(theta);
        p = [l * cos(theta_fin); 
             l * sin(theta_fin); 
             0];
    end
end

% Rotation matrix n function
function R_final = rotation_matrix_n(n, theta, alpha)
    R_final = eye(3);
    if n > 1
        for i = 1:n
            R_final = R_final * rotation_matrix(theta(i), alpha(i));
        end
    end
end

% RNEA function
function tau = rnea(q, qd, qdd, links, gravity)
    n = length(links);  % Number of links
    tau = zeros(n, 1);  % Joint torques
    
    % Preallocate arrays
    omega = zeros(n, 3);     % Angular velocity
    omegad = zeros(n, 3);    % Angular acceleration
    v = zeros(n, 3);         % Linear velocity
    vd = zeros(n, 3);        % Linear acceleration
    a_c = zeros(n, 3);
    R = cell(1, n);           % Rotation matrices
    Rbase = cell(1, n);
    p_star = cell(1, n);
    s_bar = cell(1, n);
    
    % Forward recursion
    for i = 1:n
        % Unpack link parameters
        theta = links{i}(1);
        alpha = links{i}(2);
        r = links{i}(3);
        m = links{i}(4);
        I = links{i}(5);
        j_type = links{i}(6);
        b = links{i}(7);
        
        % Compute rotation matrices and vectors
        R{i} = rotation_matrix(double(q(i)), alpha);
        Rbase{i} = rotation_matrix_to_base(q, links, i);
        p_star{i} = p_star_vector(q(i), r);
        s_bar{i} = s_vector(q(i), r);
        
        % Handle first link differently
        if i == 1
            if j_type == 1
                omega(i, :) = [0, 0, qd(i)];
                omegad(i, :) = [0, 0, qdd(i)];
                vd(i, :) = gravity';
            else
                omega(i, :) = zeros(1, 3);
                omegad(i, :) = zeros(1, 3);
                vd(i, :) = gravity' + R{i} * [qdd(i); 0; 0];
            end
        else
            % Recursive calculations for subsequent links
            if j_type == 1
                omega(i, :) = (R{i-1}' * (R{i-1} * omega(i-1, :)' + [0; 0; qd(i)]))';
                omegad(i, :) = (R{i-1}' * (Rbase{i-1} * omegad(i-1, :)' + ...
                    cross(Rbase{i-1} * omega(i-1, :)', [0; 0; qd(i)]) + [0; 0; qdd(i)]))';
                
                vd(i, :) = (omegad(i, :)' * (Rbase{i} * p_star{i}) + ...
                    cross(omega(i, :)', cross(omega(i, :)', (Rbase{i} * p_star{i}))) + ...
                    R{i-1}' * (Rbase{i-1} * vd(i-1, :)'))';
            else
                omega(i, :) = (R{i-1} * omega(i-1, :)')';
                omegad(i, :) = (R{i-1} * omegad(i-1, :)')';
                vd(i, :) = (R{i-1} * vd(i-1, :)' + R{i} * [qdd(i); 0; 0])';
            end
            
            a_c(i, :) = (omegad(i, :)' * (Rbase{i} * s_bar{i}) + ...
                cross(omega(i, :)', cross(omega(i, :)', (Rbase{i} * s_bar{i}))) + vd(i, :)')';
        end
    end
    
    % Backward recursion
    F = zeros(n, 3);
    N = zeros(n, 3);
    f = zeros(n+1, 3);
    n_torque = zeros(n+1, 3);
    
    for i = n:-1:1
        % Unpack link parameters
        theta = links{i}(1);
        alpha = links{i}(2);
        r = links{i}(3);
        m = links{i}(4);
        I = links{i}(5);
        j_type = links{i}(6);
        b = links{i}(7);
        
        if i == n
            F(i, :) = m * a_c(i, :);
            N(i, :) = Rbase{i} * I * Rbase{i}' * omegad(i, :)' + ...
                cross(omega(i, :)', Rbase{i} * I * Rbase{i}' * omega(i, :)');
            f(i, :) = F(i, :);
            n_torque(i, :) = (Rbase{i} * p_star{i} + Rbase{i} * s_bar{i})' * F(i, :)' + N(i, :)';
        else
            F(i, :) = m * a_c(i, :);
            N(i, :) = Rbase{i} * I * Rbase{i}' * omegad(i, :)' + ...
                cross(omega(i, :)', Rbase{i} * I * Rbase{i}' * omega(i, :)');
            
            f(i, :) = R{i+1} * Rbase{i+1} * f(i+1, :)' + F(i, :)';
            
            n_torque(i, :) = R{i+1} * (Rbase{i+1} * n_torque(i+1, :)' + ...
                cross(Rbase{i+1} * p_star{i}, Rbase{i+1} * f(i, :)')) + ...
                cross(Rbase{i+1} * p_star{i} + Rbase{i+1} * s_bar{i}, F(i, :)') + N(i, :)';
        end
        
        % Compute joint torque
        if j_type == 1
            tau(i) = n_torque(i, :) * (R{i-1}' * [0; 0; 1]) + b * qd(i);
        else  % Prismatic
            tau(i) = f(i, :) * (R{i-1}' * [0; 0; 1]) + b * qd(i);
        end
    end
end