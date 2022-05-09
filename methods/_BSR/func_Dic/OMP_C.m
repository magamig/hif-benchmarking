function A =OMP_C(D,X,param)
% a proximation method to solve a set of sparse coding problems.
% by solving  min_{a_i} ||a_i||_0  s.t  ||x_i-D*a_i||_2^2 <= err
%         or
%         min_{alpha} ||x-Dalpha||_2^2  s.t. ||alpha||_0 <= L
%%                    Input
% D   (mxk complex valued matrix) the dictionary
% X   (mxn complex valued matrix) the set of vector to represent by D
% X = [x_1,...,x_n];
% param     input parameter
%     param.L     the sparse parameter in solving  min_{alpha} ||x-Dalpha||_2^2  s.t. ||alpha||_0 <= L
%     param.err   the error parameter in solving   min_{a_i} ||a_i||_0  s.t. ||x_i-D*a_i||_2^2 <= err
%     ||x-Dalpha||_2^2  s.t. ||alpha||_0 <= L
%%                      output
% A   The sparse coding of each row in X according to dictionary D
[m,k] = size(D);
n = size(X,2);

if isfield(param, 'L')
    L = param.L;
else
    L = m;
end

if isfield(param, 'err')
    err = param.err;
else
    err = 0;
end

A = zeros(k,n);
Dtx  = @(x) D'*x;
for index = 1:n
    x_i = X(:,index);
    % -- Intitialize --
    % start at a_i = 0, so r_i = x_i - D*a_i = x_i
    a_i = [];
    r_i = x_i;
    err_iter = sum(abs(r_i).^2);
    numberL = 0;
    indexselect = [];
    while (numberL<L && err_iter>err)%||(isempty(indexselect))%&&~IsSubMean
        Derr = Dtx(r_i);
%         [maxvalue,maxindex] = max(abs(Derr));
        [maxvalue,maxindex] = sort(abs(Derr),'descend');
        indexselect = [indexselect,maxindex(1)];
        numberL = numberL+1;
        a_i = pinv(D(:,indexselect))*x_i;
        r_i = x_i-D(:,indexselect)*a_i;
        err_iter = sum(abs(r_i).^2);
    end
    if (~isempty(indexselect))
       A(indexselect,index)=a_i;
    end
end
