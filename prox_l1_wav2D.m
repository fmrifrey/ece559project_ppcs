function x_prox = prox_l1_wav2D(x,lam)
    
    % define soft thresholding op
    soft = @(v) sign(v).*max(abs(v) - lam, 0);

    % take the wavelet transform
    [C,S] = wavedec2(x,4,'db4');
    C_thresh = soft(C);

    % inverse wavelet transform the soft thresholded coefficients
    x_prox = waverec2(C_thresh,S,'db4');

end