function [x_star, cost_eval, tit, status] = ppgd(A,b,x0,Ax0,P,proxg,lam,cost,cost0,maxiter,maxtime,tol)
% A - fwd model
% b - measurement data
% x0 - initial solution
% Ax0 - initial foward eval
% P - preconditioner (or scalar step size)
% proxg - proximal operator [function of @(x,lam)]
% lam - regularization parameter
% cost - cost function [function of @(Ax,x)]
% cost0  - initial cost eval at x0
% maxiter - maximum number of iterations to run
% maxtime - wall time limit
% tol - convergence tolerance

    % initialize cost and iteration time
    tit = zeros(maxiter+1,1);
    cost_eval = zeros(maxiter+1,1);
    cost_eval(1) = cost0;

    % initialize solution and forward op
    x_k = x0;
    Ax_k = Ax0;

    % set status
    status = 0;

    % loop through iterations
    for k = 1:maxiter
        fprintf('ppgd step %d: cost = %.3f, elapsed time = %.fs\n', k, cost_eval(k), sum(tit));

        % start iteration timer
        t0 = tic;

        % take proximal gradient step
        x_k = proxg(x_k - P * reshape(A'*(Ax_k - b), A.idim), lam);
        Ax_k = A*x_k;

        % save iteration time and cost
        tit(k+1) = toc(t0);
        cost_eval(k+1) = cost(Ax_k,x_k);

        % determine if time limit, convergence or divergence criteria is met
        if sum(tit) > maxtime
            fprintf('wall time limit met! terminating early...\n');
            cost_eval = cost_eval(1:k+1);
            tit = tit(1:k+1);
            break
        elseif abs(cost_eval(k) - cost_eval(k+1)) / cost_eval(k) < tol
            fprintf('convergence critera met! terminating early...\n');
            status = 1;
            cost_eval = cost_eval(1:k+1);
            tit = tit(1:k+1);
            break
        elseif cost_eval(k+1) > cost_eval(1)
            fprintf('divergence detected! terminating early...\n');
            status = 2;
            cost_eval = cost_eval(1:k+1);
            tit = tit(1:k+1);
            break
        end

    end

    % return solution
    x_star = x_k;

end