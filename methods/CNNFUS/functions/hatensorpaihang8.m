function [ A h]=hatensorpaihang8(S,a,mn)

    b=find(a==mn);

h=length(b);
for i=1:length(b)
     n=ceil(b(i)/128); 
    m=b(i)-128*(n-1);
    A{i}=S(8*m-7:8*m,8*n-7:8*n,:);
end 

