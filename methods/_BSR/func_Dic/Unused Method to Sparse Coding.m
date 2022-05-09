               %Another possibility: L1 norm LASSO optimization
          %               for k=1:nb_sub
%                   %% LASSO to estimate the code
%                   Im = restoreFromSupp(conv2im(VX_dec(k,:),1), D_s(:,:,:,k), IsSubMean);
%                   XPat=im2col(conv2im(VX_dec(k,:),1),[patsize patsize],'sliding');
%                   tic;
%                   for ii=1:size(XPat,2)
%                      VXd_dec=lasso(D(:,:,k),XPat(:,ii));
%                   end
%                   toc
%                   VXd_dec(k,:) = conv2mat(Im,1);


if strcmp(method,'BbB')
          for k=1:nb_sub
              Im = restoreFromSupp(conv2im(VX_dec(k,:),1), D, supp, IsSubMean);
              VXd_dec(k,:) = conv2mat(Im,1);
          end
      elseif strcmp(method,'rotate')
          % Jose's method
          %             for k=1:size(VX_dec,1)
          %                 Im = restoreFromSupp(conv2im(VX_dec(k,:),1), D(:,:,Dic_index(k)), supp(:,:,Dic_index(k)), IsSubMean);
          %                 VXd_dec(k,:) = conv2mat(Im,1);
          %             end      
    
          %           X_restore=X_source;
          %           VXd_dec=reshape(X_restore,[size(X_restore,1)*size(X_restore,2) size(X_restore,3)])'; %use the restored image as constraint
          %           VXd_dec=VX_dec_ref;   %use the groundtruth as constraint
          
          %           temp=var_dim(XH(1:psfY.ds_r:end,1:psfY.ds_r:end,:),P_dec);
          %           X_restore=ima_interp_spline(temp,psfY.ds_r);
          
          %             Ima_DL=conv2im(VX_dec,size(VX_dec,1));
          %             Ima_DL_hat=zeros(size(Ima_DL));
          %             for i=1:size(VX_dec,1)
          %                 [Ima_DL_hat(:,:,i),~] = compCode(Ima_DL(:,:,i), D(:,:,i), IsSubMean, maxAtoms, delta, method);     
          
          % %                 alpha_tem = sunsal_simple(D(:,:,i),im2col(Ima_DL(:,:,i),[patsize patsize],'sliding'),0.11,delta);
          % %                 Ima_DL_hat(:,:,i)=depatch(D(:,:,i)*alpha_tem,patsize,Ima_DL(:,:,1)) ;
          %             end
          % %             
          %           VXd_dec=conv2mat(Ima_DL_hat,size(VX_dec,1));
          %           VXd_dec = RV'*((RV*RV')\RV)*VX_dec_ref;
      elseif strcmp(method,'ms_sub')
          Yp_temp=psfZ_s*VX_dec;
          for k=1:size(Yp_temp,1)
              Im = restoreFromSupp(conv2im(Yp_temp(k,:),1), D(:,:,k), supp(:,:,k), IsSubMean);
              VXd_dec(k,:) = conv2mat(Im,1);
          end
          %             Im = restoreFromSupp(conv2im(Yp_temp,size(Yp_temp,1)), D, supp, IsSubMean,method);
          %             VXd_dec = conv2mat(Im,nb);
      elseif strcmp(method,'Cube')            
          VXd_dec = U_restore(VX_dec,D,supp,patsize,conv2im);
          %             VXd_dec = RV'*((RV*RV')\RV)*VX_dec_ref;
          %             VXd_dec= VX_dec_ref;
      else
          for k=1:size(VX_dec,1)
              Im = restoreFromSupp(conv2im(VX_dec(k,:),1), D, supp, IsSubMean);
              VXd_dec(k,:) = conv2mat(Im,1);
              %         VXd_dec(k,:)= Z_dec_ref(k,:)
              %         figure (200);imagesc(Im);figure(gcf);drawnow;
          end
      end