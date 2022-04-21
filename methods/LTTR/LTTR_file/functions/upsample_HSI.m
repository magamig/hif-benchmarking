function Yhim_up = upsample_HSI(Yhim, downsamp_factor)
[M,N,L]=size(Yhim);
Yhim_up=zeros(M*downsamp_factor,N*downsamp_factor,L);
for i=1:M
    for j=1:N
      Yhim_up(downsamp_factor*i-downsamp_factor+1:downsamp_factor*i,downsamp_factor*j-downsamp_factor+1:downsamp_factor*j,:) = repmat(Yhim(i,j,:),[downsamp_factor downsamp_factor 1]);
    end
end

end

