function x_prox = prox_l1_wav3D(x,lam)
    
    % define soft thresholding op
    soft = @(v) sign(v).*max(abs(v) - lam, 0);

    % take the wavelet transform
    wd = wavedec3(x,4,'db4');
    wd_thresh = wd;
    for i = 1:length(wd.dec)
        wd_thresh.dec{i} = soft(wd.dec{i});
    end

    % inverse wavelet transform the soft thresholded coefficients
    x_prox = waverec3(wd_thresh);

end