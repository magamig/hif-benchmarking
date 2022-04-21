function [ A h] = paihangh( B ,a,l)
j=1;
for i=1:16*16
    if a(i)==l
       c(j)=i;
        j=j+1;
    end
end
h=length(c);
for i=1:length(c)
 m=ceil(c(i)/16); 
    n=c(i)-16*(m-1);
  
A(1:32,32*i-31:32*i,:)=B(32*m-31:32*m,32*n-31:32*n,:);

 
      
end

end


