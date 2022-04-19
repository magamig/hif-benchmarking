function  Save_HSI( msi, sz, Out_dir, fname )
nb   =   size( msi, 1 );

for  i  = 1 : nb
    im     =   reshape( msi(i,:), sz );
    str    =   strcat(fname, sprintf('_%.2d.png', i));
    imwrite( im, fullfile(Out_dir, str));     
end