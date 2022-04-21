function  [ blocks, reg_params ] = ExtractBlocks1( img, params )

%==========================================================================
% Divide a multi-spectral imagery (MSI) into blocks, each of which is a 3rd
%   order tensor with all bands and a specific spatial domain.
%
% Syntax:
%   [ blocks, reg_params ] = ExtractBlocks( img, params );
%
% Input arguments:
%   img.........the MSI tensor.(required)
%   params......an option structure whose fields are as follows:
%       block_sz: a 1-by-2 integer vector that indicates the size of blocks. 
%               If block_sz is a scalar, size of all dimensions will be the
%               same. (required)
%       overlap_sz: a 1-by-2 integer vector that indicates the size of
%               overlaps. If overlap_sz is a scalar, overlap size of all
%               dimensions will be the same.(required)
%       decouple: a logical variable. If it is true, then each spatial
%               slice of blocks is vectorized.(default false)
%
% Output arguments:
% 	blocks......the extracted block tensor. Its first and second mode
%               are y axis and x axis respectively. Its last mode indicates
%               the various blocks.
% 	reg_params..the regularized parameters including overlap_sz and
%               block_num, which may be used in function JointBlocks.
%
% See also JointBlocks
%
% by Yi Peng
%==========================================================================
block_sz=params.block_sz;
sz = size(img);
step= params.block_sz(1) -params.overlap_sz(1);
 
sz1=[1:step:sz(1)- params.block_sz(1)+1];
 sz1=[sz1 sz(1)- params.block_sz(1)+1];
sz2=[1:step:sz(2)- params.block_sz(2)+1];
sz2=[sz2 sz(2)- params.block_sz(2)+1];
block_num(1)=length(sz1);
block_num(2)=length(sz2);
blocks = zeros([block_sz, sz(3), prod(block_num)]);
for i = 1:block_num(1)
    for j = 1:block_num(2)
        ii = sz1(i);
        jj =sz2(j);
        idx = (j-1)*block_num(1) + i;
        blocks(:, :, :, idx) = ...
            img(ii:ii+block_sz(1)-1, jj:jj+block_sz(2)-1, :);
    end
end

