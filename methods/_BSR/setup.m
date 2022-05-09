%  Set the path of necessary m file.
%  Date: Oct. 2nd, 2014
%  QI WEI, University of Toulouse

clear mypath;   
st = pwd;
mypath = [ ...
    [st,';'], ...    
    [st,'\global_func;'],...
    [st,'\func_Dic;'], ... 
    [st,'\global_func\include;'],...
    ];
addpath(mypath);
clear mypath;



