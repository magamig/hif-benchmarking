function [ A f] = paili( B ,a,l)
j=1;
A=[];
for i=1:262144
    if a(i)==l
        c(j)=i;
        j=j+1;
    end
end
f=length(c);
d=1;
for i=1:length(c)
 n=ceil(c(i)/512);

    m=c(i)-512*(n-1);
   
A(:,d)=B(n,m,:);
       d=d+1;
      
end

end


