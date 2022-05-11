function [ A h]=knnpaihang8(S,a,mn)

    

h=size(a,1);
for i=1:h
     n=ceil(a(i,mn)/64); 
    m=a(i,mn)-64*(n-1);
    A{i}=S(8*m-7:8*m,8*n-7:8*n,:);
end 