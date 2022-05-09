function y_out = P_B_square(z,B_info)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  P_null_B_square
%
%  Projects onto the orthogonal complement of the nullspace of the matrix
%
%    B = [  X' \tensor I   I \tensor A  ]
%        [  CA'            0            ]
%
%  Inputs:
%     z   - mn+np dimensional vector
%
%  Outputs:
%     y - mn+np dimensional output
%
%  This function uses the assumption that the matrix A is square and
%  invertible to significantly speed up the projection onto the nullspace
%  of the constraint matrix B. Should be useful for large-scale problems. 
%
%  Winter '10, John Wright - questions? jowrig@microsoft.com
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y_out = z - P_null_B_square_new(z,B_info); 


