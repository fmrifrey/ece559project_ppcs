%% toy example - variable density sampling 2DFT
%x_tru = double(imread('cameraman.tif')); % load image
x_tru = phantom(64);
img_type = 2;

% get sizes
[Nx,Ny] = size(x_tru);
N = Nx*Ny;
M = 2*N;

% generate desired singular value spectrum
minMN = min(M,N);
r = N; % approximate rank
sigma = exp(-2*(0:minMN-1)/r);

% get true min and max eigenvalues of A^HA
AHA_maxeig = sigma(1)^2;
AHA_mineig = sigma(end)^2;

% generate random M x N
[U,~] = qr(randn(M,minMN),0);
[V,~] = qr(randn(N,minMN),0);

% forward op
A_mat = U*diag(sigma)*V';
A = fatrix2('idim', [Nx,Ny], ...
    'odim', [M,1], ...
    'forw', @(~,x) A_mat*x(:), ...
    'back', @(~,x) reshape(A_mat'*x,Nx,Ny) ...
    );

% data (inverse crime)
b = A*x_tru + 1e-2*randn(M,1);

%% estimate minimum and maximum eigenvalues of A^HA w/ power iteration
maxiter = 50;
tol = 1e-3;

% define Gram matrix linear operator
AHA = fatrix2('idim', A.idim, ...
    'odim', A.idim, ...
    'forw', @(~,x) reshape(A'*(A*x),A.idim), ...
    'back', @(~,x) reshape(A'*(A*x),A.idim) ...
    );

% calculate maximum eigenvalue of A^HA with power iteration
[~,AHA_maxeig] = pwr_itr(AHA,maxiter,tol);
AHA_maxeig = real(AHA_maxeig);

% define B = A^HA - lam_max * I
B = fatrix2('idim', A.idim, ...
    'odim', A.idim, ...
    'forw', @(~,x) reshape(AHA*x,A.idim) - AHA_maxeig*x, ...
    'back', @(~,x) reshape(AHA*x,A.idim) - AHA_maxeig*x ...
    );

% calculate minimum ev of A^HA by calculating maximimum ev of B
[~,B_maxeig] = pwr_itr(B,maxiter,tol);
AHA_mineig = real(B_maxeig + AHA_maxeig);

%% set optimization parameters
maxiter = 1e6; % max number of iterations
maxtime = 20; % wall time limit (s)
lam = 1e-3; % regularization parameter
lamp = 5e-3; % regularization parameter (preconditioned case)
tol = 1e-6; % convergence tolerance
dtests = [4 8 16]; % array of polynomial degrees for preconditioning
L = AHA_maxeig; % Lipschitz constant
if img_type == 2 % 2D
    prox = @prox_l1_wav2D;
    cost = @(Ax,x) norm(Ax - b,'fro') + lam*cost_l1_wav2D(x); % cost fun
else % 3D
    prox = @prox_l1_wav3D;
    cost = @(Ax,x) norm(Ax - b,'fro') + lam*cost_l1_wav3D(x); % cost fun
end

% create tests array
tests = repmat(struct('d', [], ...
    'cost_ppgd', [], ...
    'tit_ppgd', [], ...
    'sol_ppgd', [], ...
    'status_ppgd', []), ...
    [length(dtests),1]);

% initializiation
x0 = zeros(A.idim);
Ax0 = A*x0;
cost0 = cost(Ax0,x0);

%% solve with unpreconditioned PGD (Lipschitz step size)
[sol_pgd, cost_pgd, tit_pgd, status_pgd] = ppgd( ...
    A, ... % forward model
    b, ... % measurement data
    x0, ... % initial solution
    Ax0, ... % initial forward op
    1/L, ... % step size
    prox, ... % promixal operator
    lam, ... % regularization parameter
    cost, ... % cost function
    cost0, ... % initial cost value
    maxiter, ... % max iterations
    maxtime, ... % max wall time
    tol ... % tolerance
    );

%% solve with preconditioned PGD for different degrees of polynomial p(A^HA)
for i = 1:length(dtests)

    tests(i).d = dtests(i);

    % get preconditioner
    P = poly_precon(A,tests(i).d,AHA_maxeig,AHA_mineig);

    % PGD
    [tests(i).sol_ppgd, ...
        tests(i).cost_ppgd, ...
        tests(i).tit_ppgd, ...
        tests(i).status_ppgd] = ppgd( ...
            A, ... % forward model
            b, ... % measurement data
            x0, ... % initial solution
            Ax0, ... % initial forward op
            P, ... % step size
            prox, ... % promixal operator
            lamp, ... % regularization parameter
            cost, ... % cost function
            cost0, ... % initial cost value
            maxiter, ... % max iterations
            maxtime, ... % max wall time
            tol ... % tolerance
            );

end

%% plot results
figure

for i = 1:length(dtests)

    plot(cumsum(tests(i).tit_ppgd),tests(i).cost_ppgd,'Linewidth',2)
    hold on

end
plot(cumsum(tit_pgd),cost_pgd,'--k','linewidth',2); hold off
xlabel('wall time (s)')
ylabel('cost')
% xlim([0 2])
ylim([0.9*min(cost_pgd(:)) 1.1*cost0])

legend_entries = cell(0,0);
for i = 1:length(dtests)
    legend_entries = [legend_entries; sprintf("d = %d", tests(i).d)];
end
legend_entries = [legend_entries; "baseline (Lipschitz)"];
legend(legend_entries{:});