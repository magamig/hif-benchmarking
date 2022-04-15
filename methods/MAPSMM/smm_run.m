%% summ_run
function Out=smm_run(data,PC,npure)

%npure=4;
npure_max=npure;
nlevels=npure;
nb=npure+1;
minit=5;
sigma=0.0;
scale=1.0;
F=0.98;
niter=10;
niter2=0;
r=2;
name='Test Grobal';

% Read in data

nr=size(data,2); % xdata
nc=size(data,3); % ydata
pct=data(1:nb,:,:);
%clear data

disp(['Case number:' name]);
%disp(['Input data:' fname]);
disp(['Image size:' num2str(nr) 'x' num2str(nc)]);

% Initialize SMM

[prior,m,C,m_pure,C_pure,f,L,J,index]=smm_init(pct,npure,npure_max,nlevels,minit,sigma,scale,data,PC,F);

% Perform SMM fit iterations

[prior,m,C,m_pure,C_pure,fmap,L,J]=smm_iterate(pct,prior,m,C,m_pure,C_pure,L,J,f,index,niter);

%if niter2>0
%    
%    [prior,m,C,f,L,J,index]=smm_reset(pct,m_pure,C_pure,L,J,npure,npure_max,nlevels);
    
%    [prior,m,C,m_pure,C_pure,L,J]=smm_iterate(pct,prior,m,C,m_pure,C_pure,L,J,f,index,niter);
%end

% Complete SMM fit

[fmap,csum]=smm_finish(pct,prior,m,C,f);
Ind=max(fmap,[],1);

% Save data

Out.fmap=fmap;
Out.m_pure=m_pure;
Out.C_pure=C_pure;

% Plot result

%smm_plot(m_pure,L,J,csum,fmap);