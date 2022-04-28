function [] = demoImproveD()
% =========================================================================
%          Demo of Multiple Dictionary Updates and Coefficient Reuse
% =========================================================================
%   This file reproduces the Figures for paper:
%   "Improving Dictionary Learning: Multiple Dictionary Updates and Coefficient Reuse"
%   by Leslie N. Smith and Michael Elad, IEEE Signal Processing Letters, 
%	Vol 20, Issue 1, DOI.10.1109/LSP.2012.2229976 (2013)
%
%   Figures produced here will vary somewhat from those in the paper
%   because of the randomness in the method (i.e., initializing the
%   dictionary, dictionary training, etc.)
% =========================================================================

global ompparams exactsvd
exactsvd = 0;

% FIGURE = 1;  %  Specify which figure to create; 1, 2, or 3
for FIGURE = 1:3   %   Uncomment these lines and the "end" at the end to produce all the figures.

    if FIGURE == 1
        %  Figure 1
        imsize = 8;
        params.imsize=imsize;
        imsize2 = imsize^2;
        params.Tdata=round(imsize2/10);            % Number of target sparse vectors
        params.dictsize = 3*imsize2;      %   441;      % Number of dictionary elements (# columns of D)
        params.iternum = 30;
        fprintf(' imsize,Tdata= %g, %g,  dictsize,iternum =  %g, %g  \n', ....
            imsize,params.Tdata,params.dictsize,params.iternum);

        [Data,TestData] = buildTrainingData(imsize);
        Data = bsxfun(@minus,Data,mean(Data));  %   Remove the mean
        TestData = bsxfun(@minus,TestData,mean(TestData));  %   Remove the mean

        p = randperm(size(Data,2));
        Dinit = Data(:,p(1:params.dictsize));

        disp('DUC = 1')
        params.DUC = 1;  % Repeat dictionary update step
        size(Data)
        [D,Coeffs,err] = trainD(Dinit, Data, [],TestData, params);
        err1 = err(:,1);
        gerr1 = err(:,2);

        disp('DUC = 2')
        params.DUC = 2;  % Repeat dictionary update step
        [D,Coeffs,err] = trainD(Dinit, Data, [],TestData, params);
        err2 = err(:,1);
        gerr2 = err(:,2);

        disp('DUC = 4')
        params.DUC = 4;  % Repeat dictionary update step
        [D,Coeffs,err] = trainD(Dinit, Data, [],TestData, params);
        err4 = err(:,1);
        gerr4 = err(:,2);

        save DUCs8 params err1 err2 err4 gerr1 gerr2 gerr4 

        %   Number of multiplications for batch_OMP = 1.03*10^9
        %   Number of multiplications for optimize_atom = 6.06*10^7
        time2DUCS = 1 + 6.06/103;
        time4DUCS = 1 + 3*6.06/103;

        figure;
        pl = plot(1:params.iternum,err1,'-',[1:params.iternum]*time2DUCS,err2,'-.',[1:params.iternum]*time4DUCS,err4,'--');
        set(pl,'LineWidth',2.0,'MarkerSize',10)
        xlabel('Computations','FontSize',18,'FontName','Times');
        set(gca,'FontName','Times');
        ylabel('RMSE','FontSize',18,'FontName','Times');
        title('Representation RMSE - Training Data','FontSize',18,'FontName','Times');
        lh=legend('1 DUC','2 DUCs','4 DUCs');
        set(lh,'FontSize',18,'FontName','Times')
        set(gca,'FontSize',18,'LineWidth',2.0,'FontName','Times')


        figure;
        pl = plot(1:params.iternum,gerr1,'-',[1:params.iternum]*time2DUCS,gerr2,'-.',[1:params.iternum]*time4DUCS,gerr4,'--');
        set(pl,'LineWidth',2.0,'MarkerSize',10)
        xlabel('Computations','FontSize',18,'FontName','Times');
        set(gca,'FontName','Times');
        ylabel('RMSE','FontSize',18,'FontName','Times');
        title('Representation RMSE - Testing Data','FontSize',18,'FontName','Times');
        lh=legend('1 DUC','2 DUCs','4 DUCs');
        set(lh,'FontSize',18,'FontName','Times')
        set(gca,'FontSize',18,'LineWidth',2.0,'FontName','Times')
    elseif FIGURE == 2
        %   Figure 2; OMP + CoefROMP

        params.DUC = 1;  % Repeat dictionary update step
        params.MOD_LSX = 0;  % D, then X dictionary update
        params.iternum = 30;
        imsize = 15;
        params.imsize=imsize;
        imsize2 = imsize^2;
        params.Tdata=round(imsize2/10);            % Number of target sparse vectors
        params.card = params.Tdata;
        params.maxAtoms=round(imsize2/10);            % Number of target sparse vectors
        params.dictsize = 3*imsize2;      %   441;      % Number of dictionary elements (# columns of D)

        [Data,TestData] = buildTrainingData(imsize);
        Data = bsxfun(@minus,Data,mean(Data));  %   Remove the mean
        TestData = bsxfun(@minus,TestData,mean(TestData));  %   Remove the mean

        p = randperm(size(Data,2));
        Dinit = Data(:,p(1:params.dictsize));

        [D,Coeffs,err] = trainD(Dinit, Data, [],TestData, params);
        err1 = err(:,1);
        gerr1 = err(:,2);

        params.startRepl = 2;
        params.iterX = 2;
        params.incrAdd = 0;
        params.addK = round(imsize2/30);  % number of coefficients for CoSaMP to add each iteration
        params.sparseMeth =  'batchCoefROMP';
        [D,Coeffs,err] = trainD(Dinit, Data, [],TestData, params);
        err2 = err(:,1);
        gerr2 = err(:,2);

        save CoROMPDiX15 params err1 err2  gerr1 gerr2  
        
        % dim=225;nAtoms=675;k=23;nSignals=50000; avgIter=5.8;k3=round(k/3);
        % M_batch_CoefROMP=dim*nAtoms^2 +nSignals * (dim*nAtoms +k3*(1+nAtoms +k3^2) + avgIter*k*(2*k^2 +nAtoms+1) );
        % M_batch_CoefROMP= 1.955804562500000e+10
        % M_batch_OMP=dim*nAtoms^2 + nSignals * [ dim * ( 1 + nAtoms) + k^2 * ( 1 + nAtoms + k^2 ) ];
        % M_batch_OMP=3.957976562500000e+10
        % m_D_update = nAtoms * [ 2  + dim * ( 3 + nAtoms ) + avgAtoms * ( 2 + nAtoms) ]
        % M_Total_OMP =    4.0461e+10
        % M_Total_CoefROMP =   2.0440e+10

        timeCoefROMP = 2.044 / 4.0461;
        figure;
        pl = plot(1:params.iternum,err1,'-',[1:params.iternum]*timeCoefROMP,err2,'-.');
        set(pl,'LineWidth',2.0,'MarkerSize',10)
        xlabel('Computations','FontSize',18,'FontName','Times');
        ylabel('RMSE','FontSize',18,'FontName','Times');
        title('Representation RMSE - Training Data','FontSize',18,'FontName','Times');
        lh=legend('OMP','CoefROMP');
        set(lh,'FontSize',18,'FontName','Times')
        set(gca,'FontSize',18,'LineWidth',2.0,'FontName','Times')

        figure;
        pl = plot(1:params.iternum,gerr1,'-',[1:params.iternum]*timeCoefROMP,gerr2,'-.');
        set(pl,'LineWidth',2.0,'MarkerSize',10)
        xlabel('Computations','FontSize',18,'FontName','Times');
        ylabel('RMSE','FontSize',18,'FontName','Times');
        title('Representation RMSE - Testing Data','FontSize',18,'FontName','Times');
        lh=legend('OMP','CoefROMP');
        set(lh,'FontSize',18,'FontName','Times')
        set(gca,'FontSize',18,'LineWidth',2.0,'FontName','Times')

    elseif FIGURE == 3
        %   Figure 3; Denoise OMP vs CoefROMP

        %   Read in image
        img = 1;
        imgLibrary={'barbara','lena','boat','house','peppers256','fingerprint','flinstones'};
        pathForImages ='./images/';
        IMin0 = double(imread(strcat([pathForImages,imgLibrary{img},'.png'])));
        if (length(size(IMin0))>2)
            IMin0 = rgb2gray(IMin0);
        end
        if (max(IMin0(:))<2)
            IMin0 = IMin0*255;
        end
        disp(imgLibrary{img})

        %   Figure 3a; 8x8 case
        bb=8; % block size
        nIters = 12;
        sigma = 25; 
        psnr = denoiseFig(bb,IMin0,nIters,sigma,0.55);
        save iterDUC8-1-25  nIters psnr

        figure;
        pl = plot(1:nIters,psnr(:,1,2),'-+',1:nIters,psnr(:,2,2),'-o', ...
        1:nIters,psnr(:,1,1),'-.+',1:nIters,psnr(:,2,1),'-.o');
        set(pl,'LineWidth',2.0,'MarkerSize',10);
        ylabel('PSNR','FontSize',18,'FontName','Times'); 
        xlabel('Iteration','FontSize',18,'FontName','times');
        title('Denoising Performance for 8\times 8 Patches','FontSize',18,'FontName','Times');
        lh = legend('CoefROMP; 1 DUC','CoefROMP; 2 DUC','OMP; 1 DUC','OMP; 2 DUC',4);
        set(lh,'FontSize',14)
        set(gca,'FontSize',18,'LineWidth',2.0,'FontName','Times')

        %   Figure 3b; 16x16 case
        bb = 16; % block size

        psnr = denoiseFig(bb,IMin0,nIters,sigma,0.25);
        save iterDUC16-1-25 nIters psnr

        figure; 
        pl = plot(1:nIters,psnr(:,1,2),'-+',1:nIters,psnr(:,2,2),'-o', ...
            1:nIters,psnr(:,1,1),'-.+',1:nIters,psnr(:,2,1),'-.o');
        set(pl,'LineWidth',2.0,'MarkerSize',10);
        ylabel('PSNR','FontSize',18,'FontName','Times'); 
        xlabel('Iteration','FontSize',18,'FontName','times');
        title('Denoising Performance for 16\times 16 Patches','FontSize',18,'FontName','Times');
        lh = legend('CoefROMP; 1 DUC','CoefROMP; 2 DUC','OMP; 1 DUC','OMP; 2 DUC',4);
        set(lh,'FontSize',14)
        set(gca,'FontSize',18,'LineWidth',2.0,'FontName','Times')
    end

end

end

function [Data,TestData] = buildTrainingData(imsize)

    directory = './images/';
    [contents]=dir(directory);
    TrainSamples = 6000;
    NImgs = length(contents)-2;
    
    for j=1:NImgs
%         disp([directory,contents(j+2).name]);
        x=double(imread([directory,contents(j+2).name]));
        d = im2col(x,[imsize imsize],'sliding');
        [~,indx] = sort(std(d),'descend');
        data(:,(j-1)*TrainSamples+1:j*TrainSamples) = d(:,indx(1:TrainSamples));
    end
    data = bsxfun(@minus,data,mean(data));
    p = randperm(NImgs*TrainSamples);
    Data = data(:,p(1:50000));
    TestData = data(:,p(50001:60000));

end

function psnr = denoiseFig(bb,IMin0,nIters,sigma,coeffFact)
    err = 1.1;
    addK = max(1,round(bb^2/60)); 
    addX = 1.0;
    slide = 1;
    fprintf('=> sigma = %g   slide = %g  ',sigma,slide);
    IMin=IMin0+sigma*randn(size(IMin0));
    PSNRIn = 20*log10(255/sqrt(mean((IMin(:)-IMin0(:)).^2)));
    fprintf(' PSNRin= %g \n',PSNRIn);
    nAtoms = 4*bb^2; 
        
    rng_DUC = [1 2];
    len_DUC = length(rng_DUC);
    rng_m = [1 3];
    psnr = zeros(nIters,len_DUC,2);
    for meth=1:2
        for d = 1:len_DUC
            DUC=rng_DUC(d);
            fprintf('=> DUC = %g  \n',DUC);
            initD = [];
            CoefMatrix = [];
            for iter=1:nIters
                [IoutAdaptive,output,CoefMatrix] = denoiseImageKSVD(IMin, sigma,nAtoms,initD,CoefMatrix, ...
                    'numKSVDIters',1,'errorFactor',err,'errorFlag',rng_m(meth),'DUC',DUC,'waitBarOn',0, ...
                    'addK',addK,'addX',addX,'coeffFact',coeffFact,'blockSize',bb,'slidingFactor',slide);
                initD = output.D;

                PSNROut = 20*log10(255/sqrt(mean((IoutAdaptive(:)-IMin0(:)).^2)));
                fprintf(' =>iter,PSNROut= %g, %g \n',iter,PSNROut);
                psnr(iter,d,meth) = PSNROut;
            end
            fprintf('===>RESULT: nAtoms,Method,DUC,PSNROut= %g, %g, %g, %g \n',nAtoms,meth,DUC,PSNROut);
        end
    end
end