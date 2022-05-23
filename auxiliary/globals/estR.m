function [R,error] = estR(HS,MS,mask,Pre_R)
%--------------------------------------------------------------------------
% Estimation of relative spectral response functions (SRFs)
% 
% Estimate relative SRFs via quadratic programming
%
% USAGE
%       R = estR(HS,MS,mask,Pre_R)
%
% INPUT
%       HS   : Low-spatial-resolution HS image (rows2,cols2,bands2)
%       MS   : MS image (rows1,cols1,bands1)
%       mask : (optional) Binary mask for processing (rows2,cols2) (mainly
%             for real data)
%       Pre_R: (optional) Pre-launch SRFs (only limit spectral ranges)
%
% OUTPUT
%       R    : Relative SRFs 
%              (bands1,bands2+1) (consider offset)
%       error: Reconstruction errors for all MS bands (bands1)
%
% AUTHOR
% Naoto Yokoya, University of Tokyo
% Email: yokoya@sal.rcast.u-tokyo.ac.jp
%--------------------------------------------------------------------------

[rows1,cols1,bands1] = size(MS);
[rows2,cols2,bands2] = size(HS);
if nargin == 2
    mask = ones(rows2,cols2);
end

HS = reshape([reshape(HS,[],bands2) reshape(mask,[],1)],rows2,cols2,[]);
bands2 = size(HS,3);

R = zeros(bands1,bands2);

% downgrade spatial resolution
% LR_MS = zeros(rows2,cols2,bands1);
% for b = 1:bands1
%     LR_MS(:,:,b) = imresize(reshape(MS(:,:,b),rows1,cols1),[rows2 cols2]);
% end
% LR_MS(LR_MS<0) = 0;
LR_MS = gaussian_down_sample(MS,rows1/rows2);
A = reshape(HS,[],bands2);

H = A'*A;
options = optimset('Algorithm','interior-point-convex','Display','off','MaxIter',500);

if nargin < 4
    error = zeros(bands1,1);
    for k = 1:bands1
        b = double(reshape(LR_MS(:,:,k),[],1));
        f = -A'*b;
        C = -eye(bands2);
        C(end,end) = 0;
        e = zeros(bands2,1);    
        x = quadprog(H,f,C,e,[],[],[],[],[],options);
        R(k,:) = reshape(x,1,[]);
        error(k,1) = mean((b(mask==1)-A(mask==1,:)*x).^2).^0.5/mean(b(mask==1));
    end 
else
    for k = 1:bands1
        b = double(reshape(LR_MS(:,:,k),[],1));
        A2 = A(:,Pre_R(k,:)>0);
        H2 = A2'*A2;
        f = -A2'*b;
        C = -eye(sum(Pre_R(k,:)>0));
        C(end,end) = 0;
        e = zeros(sum(Pre_R(k,:)>0),1);    
        x = quadprog(H2,f,C,e,[],[],[],[],[],options);
        R(k,Pre_R(k,:)>0) = reshape(x,1,[]);
    end
end
