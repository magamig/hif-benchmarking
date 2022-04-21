
function  [X] =  WNNM( Y, C, NSig)
    [U,SigmaY,V] = svd(full(Y),'econ');
     SigmaY=diag(SigmaY);
    X=zeros(size(Y));
%     PatNum       = size(Y,2);
%     TempC  = C*sqrt(PatNum)*2*NSig^2;
 TempC  = C*NSig^2;
    [SigmaX,svp] = ClosedWNNM(SigmaY,TempC,eps); 
    if svp>=1
    X =  U(:,1:svp)*diag(SigmaX)*V(:,1:svp)';   
    end
return;
