function [lx, ux] = LP_solver(C, V, Ca , da, lb, ub)
    
dim = size(C,1);
% Define the objective function coefficients
% Since x = C + V' * alpha, we optimize over alpha with coefficients V'

lx = zeros(dim,1);
ux = zeros(dim,1);

C = double(C);
V = double(V);
if ~isempty(Ca)
    Ca = double(Ca);
    da = double(da);
end
lb = double(lb);
ub = double(ub);

% options = optimoptions('linprog', 'Display', 'none');
options = mskoptimset('Display', 'off');

rem = mod(dim , 1000);
t = 0;
for i=1:floor(dim/1000)
    myV = full(V(t+1:t+1000, :));
    myC = C(t+1:t+1000, :);
    mylx = zeros(1000,1);
    myux = zeros(1000,1);
    disp(['processing elements ' num2str(t+1) ' through ' num2str(t+1000) '.'])
    parfor j=1:1000
        [~,fval_min,~,~,~] = linprog( myV(j, :)',Ca,da,[],[],lb,ub,rand(dim,1),options);
        [~,fval_max,~,~,~] = linprog(-myV(j, :)',Ca,da,[],[],lb,ub,rand(dim,1),options);

        % Extract optimal values
        mylx(j,1) = myC(j,1) + fval_min;
        myux(j,1) = myC(j,1) - fval_max;
    end
    lx(t+1:t+1000,1) = mylx;
    ux(t+1:t+1000,1) = myux;
    t = t+1000;

end
disp(['processing elements ' num2str(t+1) ' through ' num2str(t+rem) '.'])

myV = full(V(t+1:t+rem, :));
myC = C(t+1:t+rem, :);
mylx = zeros(rem,1);
myux = zeros(rem,1);

parfor j=1:rem

    [~,fval_min,~,~,~] = linprog( myV(j, :)',Ca,da,[],[],lb,ub,rand(dim,1),options);
    [~,fval_max,~,~,~] = linprog(-myV(j, :)',Ca,da,[],[],lb,ub,rand(dim,1),options);

    % Extract optimal values
    mylx(j,1) = myC(j,1) + fval_min;
    myux(j,1) = myC(j,1) - fval_max;

end
lx(t+1:t+rem,1) = mylx;
ux(t+1:t+rem,1) = myux;

end