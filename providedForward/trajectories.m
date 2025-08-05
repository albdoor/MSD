% number of links:
n = 2;                          % edit this
param.n = n;
% vectors of masses m, inertias J, link lengths l, and mass locations r:
param.m = ones(param.n,1);      % edit these
param.J = zeros(param.n,1);
param.l = ones(param.n,1);      % edit these
param.r = param.l;              % edit these
param.g = 9.81;

% initial conditions (in radians), link angles are counted from the
% vertical, positive is counterclockwise
thetas = [pi/2, pi/2]';   % edit these
dthetas = [0, 0]';        % edit these
vars0 = [thetas; dthetas];

time_f = 10;                    % edit this
time_step = 0.01;               % edit this
opts = odeset("AbsTol",1e-9,"RelTol",1e-7);

[t,vars1] = ode45(@(t,vrs) rl_absolute_ode(t,vrs,param),0:time_step:time_f,vars0,opts);

% vars1 is a (time_f/time_step) by 2*n matrix. Each row corresponds to a
% certain moment of time, while each column is the current link angle (or velocity).
% For example, for a 4-link system, 3rd column stores the angles of 3rd
% link, while the 7th column stores the angular velocity of 3rd link.

x_pos = zeros(length(vars1),n);
y_pos = zeros(length(vars1),n);

for ind = 1:length(vars1)
    for i = 1:n
        x_pos(ind,i) = sum(param.l(1:i)'.*sin(vars1(ind,1:i)));
        y_pos(ind,i) = -sum(param.l(1:i)'.*cos(vars1(ind,1:i)));
    end
end
figure()
axis equal
hold on
for i = 1:n
    plot(x_pos(:,i),y_pos(:,i))
end
xx = @(ind,n) x_pos(ind,n);
yy = @(ind,n) y_pos(ind,n);
figure()
for ind = 1:length(vars1)
    hold on; grid on;
    cla
    time = (ind-1)*0.01;
    plot([0 xx(ind,1)],[0 yy(ind,1)],'r-');
    for i = 1:n
        plot(xx(ind,i),yy(ind,i),'ro','MarkerSize',2,'MarkerFaceColor','r');
        if i<n
            plot([xx(ind,i) xx(ind,i+1)],[yy(ind,i) yy(ind,i+1)],'r-');
        end
    end
    for i = 1:n
        plot(x_pos((1:ind),i),y_pos((1:ind),i))
    end
    axis([-sum(param.l) sum(param.l) -sum(param.l) sum(param.l)])
    % axis equal
    title(time);
    drawnow
    %pause(0.01);
end


% Extract theta and dtheta
theta = vars1(:, 1:n);
dtheta = vars1(:, n+1:2*n);

% Preallocate ddtheta
ddtheta = zeros(length(t), n);

% Compute ddtheta at each time step
for i = 1:length(t)
    dd = rl_absolute_ode(t(i), vars1(i,:)', param); % Note: vars must be a column vector
    ddtheta(i, :) = dd(n+1:end)';
end

% Combine all into one matrix: [theta, dtheta, ddtheta]
results = [theta, dtheta, ddtheta];

% Define column names
colNames = [ ...
    arrayfun(@(i) sprintf("theta_%d", i), 1:n, 'UniformOutput', false), ...
    arrayfun(@(i) sprintf("dtheta_%d", i), 1:n, 'UniformOutput', false), ...
    arrayfun(@(i) sprintf("ddtheta_%d", i), 1:n, 'UniformOutput', false)];

% Create a table
resultsTable = array2table(results);

% Export to CSV
writetable(resultsTable, 'rl_multilink_simulation.csv');

