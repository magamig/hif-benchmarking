function [ind ] = L1_distance( A,k )
D=zeros(size(A,2),size(A,2));
for i=1:size(A,2)
    for j=1:size(A,2)
D(i,j)=norm(A(:,i)-A(:,j),1);
    end
end
 [foo, ind] = sort(D, 1);
          ind=ind(1:k,:);
