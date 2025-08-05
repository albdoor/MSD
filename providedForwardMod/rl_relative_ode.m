function dvrs = rl_relative_ode(t,vars,param)
    n = param.n; g = param.g;
    m = param.m; l = param.l; r = param.r;
    M = zeros(n);
    C = zeros(n);
    G = zeros(n,1);
    J = zeros(n);
    %tau = [2 * sin(0.5 * t), 1.5 * cos(1.5 * t)]';
    for i=1:n
        for j=1:n
            mksum = 0;
            a=max(i,j); b=min(i,j);
            for k=a:n
                lpsq = 0;
                for p=a:k-1
                    lpsq=lpsq+l(p)^2;
                end
                twolplqrk = 0;
                for p=a:k-1
                    lq = 0;
                    for q=p+1:k-1
                        lq=lq+l(q)*cos(sum(vars(p+1:q)));
                    end
                    twolplqrk = twolplqrk+2*l(p)*(lq+r(k)*cos(sum(vars(p+1:k))));
                end
                lplqrk = 0;
                for p=b:a-1
                    lq=0;
                    for q=a:k-1
                        lq=lq+l(q)*cos(sum(vars(p+1:q)));
                    end
                    lplqrk = lplqrk + l(p)*(lq+r(k)*cos(sum(vars(p+1:k))));
                end
                mksum = mksum + m(k)*(lpsq+r(k)^2+twolplqrk+lplqrk);
            end
            M(i,j) = mksum;
            J(i,j) = sum(param.J(a:end));
        end
    end
    for i=1:n
        for j=1:n
            a=max(i,j); b=min(i,j);
            if i==j
                mksum = 0;
                for k=a:n
                    lplqrk = 0;
                    for p=i:k-1
                        lq = 0;
                        for q=p+1:k-1
                            lq=lq+l(q)*sum(vars(n+p+1:n+q))*sin(sum(vars(p+1:q)));
                        end
                        lplqrk = lplqrk+l(p)*(lq+r(k)*sum(vars(n+p+1:n+k))*sin(sum(vars(p+1:k))));
                    end
                    mksum = mksum + m(k)*lplqrk;
                end
                C(i,j) = -2*mksum;
            end
            if i<j
                mksum = 0;
                for k=j:n
                    lplqrk = 0;
                    for p=i:j-1
                        lq = 0;
                        for q=j:k-1
                            lq=lq+l(q)*sum(vars(n+p+1:n+q))*sin(sum(vars(p+1:q)));
                        end
                        lplqrk = lplqrk+l(p)*(lq+r(k)*sum(vars(n+p+1:n+k))*sin(sum(vars(p+1:k))));
                    end
                    twolplqrk = 0;
                    for p=j:k-1
                        lq=0;
                        for q=j+1:k-1
                            lq=lq+l(q)*sum(vars(n+p+1:n+q))*sin(sum(vars(p+1:q)));
                        end
                        twolplqrk = twolplqrk + 2*l(p)*(lq+r(k)*sum(vars(n+p+1:n+k))*sin(sum(vars(p+1:k))));
                    end
                    mksum = mksum - m(k)*(twolplqrk+lplqrk);
                end
                C(i,j) = mksum;
            end
            if i>j
                mksum = 0;
                for k=i:n
                    lplqrk = 0;
                    for p=j:i-1
                        lq = 0;
                        for q=i:k-1
                            lq=lq+l(q)*sum(vars(n+1:n+p))*sin(sum(vars(p+1:q)));
                        end
                        lplqrk = lplqrk+l(p)*(lq+r(k)*sum(vars(n+1:n+p))*sin(sum(vars(p+1:k))));
                    end
                    twolplqrk = 0;
                    for p=i:k-1
                        lq=0;
                        for q=i+1:k-1
                            lq=lq+l(q)*sum(vars(n+p+1:n+q))*sin(sum(vars(p+1:q)));
                        end
                        twolplqrk = twolplqrk + 2*l(p)*(lq+r(k)*sum(vars(n+p+1:n+k))*sin(sum(vars(p+1:k))));
                    end
                    mksum = mksum + m(k)*(lplqrk-twolplqrk);
                end
                C(i,j) = mksum;
            end
        end
    end
    for i=1:n
        mksum = 0;
        for k=i:n
            lpsum = 0;
            for p=i:k-1
                lpsum = lpsum + l(p)*sin(sum(vars(1:p)));
            end
            mksum = mksum + m(k)*(lpsum + r(k)*sin(sum(vars(1:k))));
        end
        G(i) = param.g*mksum;
    end
    temp = -C*vars(n+1:2*n) - G;
    theta_dotdot = (M+J)\temp;
    dvrs = cat(1,vars(n+1:2*n),theta_dotdot);
end