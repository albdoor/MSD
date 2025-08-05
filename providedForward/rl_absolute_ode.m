function dvrs = rl_absolute_ode(t,vars,param)
    n = param.n; g = param.g; J = diag(param.J);
    m = param.m; l = param.l; r = param.r;
    M = zeros(n);
    C = zeros(n);
    G = zeros(n,1);
    lam = @(ind,k) l(ind)*(k>ind) + r(ind)*(1-(k>ind));
    tau = [2 * sin(0.5 * t), 1.5 * cos(1.5 * t)]';
    for i=1:n
        for j=1:n
            mksum = 0;
            for k=max(i,j):n
                mksum = mksum + m(k)*lam(i,k)*lam(j,k);
            end
            M(i,j) = mksum*cos(vars(i)-vars(j));
            C(i,j) = mksum*sin(vars(i)-vars(j))*vars(n+j);
        end
        mksum = 0;
        for k=i:n
            mksum = mksum + g*m(k)*lam(i,k)*sin(vars(i));
        end
        G(i) = mksum;
    end
    temp = -C*vars(n+1:2*n) - G;
    theta_dotdot = (M+J)\temp;
    dvrs = cat(1,vars(n+1:2*n),theta_dotdot);
end