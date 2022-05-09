function [M,best_permutv] = best_permut_R(M_in,M_true)
L=size(M_true,2);
best_permutv=zeros(1,L);
index_in=1:L;
M_in_com=M_in;
for i=1:L
    corr=zeros(1,size(M_in,2));
    % Find the most similar endmember for the ith endmember
     for j=1:size(M_in,2)
         tem=corrcoef(M_in(:,j),M_true(:,i));
         corr(j)=tem(2,1); % take the coefficient
     end
     [~,indx]=max(corr);
     best_permutv(i)=index_in(indx);     % put the position of this selected vector to best_permutv
     index_in(indx)=[];
     M_in(:,indx)=[];  % remove this column
end
M = M_in_com(:,best_permutv);