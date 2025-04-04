% Simple Pendulum Simulation

% Physical parameters
global g ell m
g = 9.81;   % gravitational acceleration (m/s^2)
ell = 1;    % pendulum length (m)
m = 1;      % mass (kg)

% Initial conditions
theta0 = deg2rad(30);   % initial angle (radians)
theta_dot0 = 0;         % initial angular velocity (rad/s)

% Time vector
t_span = [0 10];
t_eval = linspace(0, 10, 1000);

% Solve ODE using ode45
options = odeset('RelTol', 1e-8, 'AbsTol', 1e-8);
[t, sol] = ode45(@pendulum_ODE, t_span, [theta0, theta_dot0], options);

% Calculate acceleration
theta_ddot = zeros(size(t));
for i = 1:length(t)
    theta_ddot(i) = g * sin(sol(i,1)) / ell + 2 * sin(0.5 * t(i)) / (m * ell^2);
end

% Convert to degrees for plotting
theta_deg = rad2deg(sol(:,1));
theta_dot_deg = rad2deg(sol(:,2));

% Save data
save('simplePendLE.mat', 'sol', 'theta_ddot', 't');

% Plotting
figure;
plot(t, sol(:,1), 'r', 'LineWidth', 2);
hold on;
plot(t, sol(:,2), 'b', 'LineWidth', 2);
plot(t, theta_ddot, 'g', 'LineWidth', 2);

title('Simple Pendulum');
xlabel('Time (seconds)');
ylabel('\theta (rad), \dot\theta (rad/s), \ddot\theta (rad/s^2)');
legend('\theta', '\dot\theta', '\ddot\theta');
grid on;

% Pendulum ODE function (must be at the end of the script)
function dy = pendulum_ODE(t, y)
    global g ell m
    % y(1) = theta, y(2) = theta_dot
    dy = zeros(2,1);
    dy(1) = y(2);
    dy(2) = g * sin(y(1)) / ell + 2 * sin(0.5 * t) / (m * ell^2);
end