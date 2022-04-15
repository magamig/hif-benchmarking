function p=distribute(m,c,v)

l=0;
r=[];
t=[];
p=[];

% Check for input errors and return null result on error

if c<0
    disp('Error (distribute):');
    disp('Number of bins (c) less than zero.');
    return;
elseif m<0
    disp('Error (distribute):');
    disp('Number of items (m) less than zero.');
    return;
elseif v<0
    disp('Error (distribute):');
    disp('Bin maximum (v) less than zero.');
    return;
elseif v>c
    disp('Error (distribute):');
    disp('Bin maximum (v) greater than number of bins (c).');
    return;
end

% Recursively generate all unique, ordered combinations of item
% distributions

r=rcalc(l,m,c,v,r,t);
mr=size(r,1);

% Permute each combination and remove redundancies through sorting

for k=1:mr
    s=perms(r(k,:));
    s=sortrows(s);
    ss=sum(abs(diff(s)),2);
    I=find(ss==0);
    s(I,:)=[];
    p=[p;s];
end

return;

%%%%%

function r=rcalc(l,m,c,v,r,t)

l=l+1;
n=m-sum(t);

if l<v && n>0
    if isempty(t)
        w=inf;
    else
        w=t(end);
    end
    for k=ceil(n/(v-l+1)):min([n,w])
        r=rcalc(l,m,c,v,r,[t,k]);
    end
elseif l==v && n>0
    r=[r;[t,n,zeros([1,c-v])]];
else
    r=[r;[t,zeros([1,c-l+1])]];
end

return