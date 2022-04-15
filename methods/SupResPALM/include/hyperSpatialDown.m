function [G, Gt] = hyperSpatialDown(h,w,dsf)

% w=512/4
% h=512/4

% dsf=8;

Nm = w*h;

aux = repmat([1:dsf]',1,dsf);
aux(h,w)=0;
aux = hyperConvert2d(aux);
[~,b,c] = find(aux);


b2=[];c2=[];
for i = 0:h/dsf-1
    b2(end+1:end+length(b)) = b+i*dsf;
    c2(end+1:end+length(c)) = i+1;
end


b3=[];c3=[];
for i=0:w/dsf-1
    b3(end+1:end+length(b2)) = b2 + i*h * dsf;
    c3(end+1:end+length(c2)) = c2 + i*h/dsf;
end

G = sparse(b3,c3,1/dsf^2,Nm,Nm/dsf^2);
Gt = G'*length(b);

