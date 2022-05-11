function [ b ] = findb( b,kkk,m7,Y_h)
flag=0;
   if kkk>=2
   for j=1:kkk-1
       if m7{kkk}==m7{j}
           flag=1;
           break
       end
   end
    end
   if flag==0
    kk= Y_h(m7{kkk}(1),m7{kkk}(2),:);
    kk=kk(:);
     b=[b kk];
   end
        



