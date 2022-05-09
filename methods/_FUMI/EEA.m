function [E] = EEA(E,Uh,Um,VXH,VXM,R,ChInv,CmInv,H)
%% Initialization
mu = size(VXH,2);
Th_ADMM = 1e-4; %% Threshold to stop
Th_ADMM_dual = 1e-4;
res_Dual=inf;
N_max=5000;
subspace=1;
% V1=E*Uh;G1=V1;
% V2=E*Um;G2=V2;
% % V3=E;G3=V3;
V=1*sqrt(ChInv)*E;
G=V*0;
%% Loop
% % Tem1=(ChInv+mu*eye(size(ChInv)))\eye(size(ChInv));
% % Tem2=(R'*CmInv*R+mu*eye(size(ChInv)))\eye(size(ChInv));
% Tem1=inv(ChInv+mu*eye(size(ChInv)));
% Tem2=inv(R'*CmInv*R+mu*eye(size(ChInv)));
% wVXH=ChInv*VXH;
% wVXM=R'*CmInv*VXM;
[N_band,L]=size(E);
invUm2=eye(L)/(Um*Um');
DisM=L/(L-1)*eye(L)-1/(L-1)*ones(L,L);
lambda=0e-4*mean(diag(ChInv))*size(VXH,2); % if lambda==0, there is no minimum volume constraint
MatB=(Uh*Uh'+mu*eye(L))*invUm2;   
MatD=lambda*(DisM*DisM')*invUm2;
if subspace==0
    MatE=inv(ChInv);
    MatA=ChInv\R'*CmInv*R;  
    temp1=ChInv\R'*CmInv*VXM*Um'+VXH*Uh';
    Mat_temp=kron(eye(L),MatA)+kron(MatB',eye(N_band))+kron(MatD',MatE);  
    temp=Mat_temp\eye(N_band*L);
elseif subspace==1
    % Subspace
    RH=R*H;
    MatE=(H'*ChInv*H)\eye(L);
    MatA=MatE*((RH)'*CmInv*RH);
    temp1=MatE*H'*(ChInv*VXH*Uh'+R'*CmInv*VXM*Um');
    temp2=mu*MatE*H'*sqrt(ChInv);
    Mat_temp=kron(eye(L),MatA)+kron(MatB',eye(L))+kron(MatD',MatE);  
    temp=Mat_temp\eye(L*L);
end

upper_V=repmat(diag(sqrt(ChInv)),[1 L]);
%     [VB,DB] =eig(MatB);
%     [VA,DA] =eig(MatA); %DA(abs(DA)<1e-15)=0;
%     display(cond(VA));

%     temp2=(Um*Um')\VB;
%     AA=repmat(diag(DA),[1 L]);
%     BB=repmat(diag(DB)',[N_band 1]);
%     CoeM=real(AA+BB);
%     cost_old=norm(sqrt(ChInv)*(VXH-E*Uh),'fro')^2/2+norm(sqrt(CmInv)*(VXM-R*E*Um),'fro')^2/2;

%% Define the regularzier
for i=1:N_max
    % ADMM to uodate the endmembers
%     E = ((V1+G1)*Uh'+(V2+G2)*Um')/(Uh*Uh'+Um*Um');
% %     E = max(E ,0);
% %     E = ((V1+G1)*Uh'+(V2+G2)*Um'+(V3+G3))/(Uh*Uh'+Um*Um');
%     
%     nu1= E*Uh-G1; % Update nu1
%     nu2= E*Um-G2; % Update nu2
% %     nu3=  E  -G3; % Update nu3
%     
%     V1 = Tem1*(wVXH+mu*nu1);     % Update V1
%     V2 = Tem2*(wVXM+mu*nu2);  % Update V2
% %     V3 = min(max(nu3,0),1);          % Update V3
%     
%     G1 = V1-nu1; % Update G1
%     G2 = V2-nu2; % Update G2
% %     G3 = V3-nu3; % Update G3   
    %% Sylvester equation
%         E=(temp1+mu*(V+G))/(Uh*Uh'+mu*eye(L));
%         cost=norm(sqrt(ChInv)*(VXH-E*Uh),'fro')^2/2;
%         MatC=VA\(temp1+(sqrt(ChInv)\(V+G)))*temp2;
%         E=VA*(MatC./CoeM)/VB; 
    if subspace==0
        MatC=(temp1+mu*(sqrt(ChInv)\(V+G)))*invUm2;           
        E=reshape(temp*MatC(:),[N_band L]);
    else
        MatC=(temp1+temp2*(V+G))*invUm2;
        E=H*reshape(temp*MatC(:),[L L]);
    end
%         cost=norm(sqrt(ChInv)*(VXH-E*Uh),'fro')^2/2+norm(sqrt(CmInv)*(VXM-R*E*Um),'fro')^2/2;
    nu=sqrt(ChInv)*E-G;
    V=min(max(nu,0),upper_V);
    G=V-nu;
    %% Stop the ADMM if the residual is small enough: sqrt(ChInv)*E-V
%     norm(sqrt(ChInv)*E-V,'fro')/numel(V)
    res_Pri=sqrt(ChInv)*E-V;
    if i>1 
        V_old = V;
        res_Dual=mu*(V-V_old);
    end    
    if norm(res_Pri,'fro')/sqrt(numel(res_Pri))< Th_ADMM %&& norm(res_Dual,'fro')/sqrt(numel(res_Dual))< Th_ADMM_dual  %(abs((cost-cost_old))/cost_old < Th || i == N_max) %&& i>2
%        disp(['EndEst converges at the ' num2str(i) 'th iteration']);
        break;  
    end
%     cost_old =cost;
end
% E = min(max(E,0),1);