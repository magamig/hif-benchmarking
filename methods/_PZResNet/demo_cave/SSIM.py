def SSIM(x,y):
    meanx = x.mean(-1,True).mean(-2,True).mean(0,True)
    meany = y.mean(-1,True).mean(-2,True).mean(0,True)
    sigx = x-meanx
    sigx = sigx.mean(-1,True).mean(-2,True).mean(0,True)
    sigy = y-meany
    sigy = sigy.mean(-1,True).mean(-2,True).mean(0,True)
    SSIM = (2*meanx*meany+1e-4)*(sigx*sigy+9e-4)/((meanx**2+meany**2+1e-4)*(sigx**2+sigy**2+9e-4))
    return SSIM