function [ sam] = SAM( a1, a2 )
[m,n,l]=size(a1);

k=0;
a1=Unfold(a1,size(a1),3);
a2=Unfold(a2,size(a2),3);
b=any(a1);
a1=a1(:,b);
a2=a2(:,b);
c=any(a2);
a1=a1(:,c);
a2=a2(:,c);
a1=normc(a1);
a2=normc(a2);
a3=a1.*a2;
% a3=abs(a3);
a3=sum(a3);
sam=mean(rad2deg(acos(a3)));
% k=0
% sam=0
%  for i=1:m
% for j=1:n
%   a=a1(i,j,:);
%     b=a2(i,j,:);
%     a=a(:);
%     b=b(:);
%     if norm(a)>0 && norm(b)>0
% t=dot(a,b)/norm(a,2)/norm(b,2);
% k=k+1;
% sam=sam+rad2deg(acos(t));
%     end
% end
% end
% sam=sam/k;