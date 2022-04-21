function [ A ] = pailie( B ,a,l)
j=1;
for i=1:16*16
    if a(i)==l
       c(j)=i;
        j=j+1;
    end
end
d=1;
for i=1:length(c)
 n=ceil(c(i)/16);
  m=c(i)-16*(n-1);
   for e=32*n-31:32*n
      for f=32*m-31:32*m
A(:,d)=B(f,e,:);
       d=d+1;
      end
   end
end

end

