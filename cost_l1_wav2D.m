function cost = cost_l1_wav2D(x)

    % take the wavelet transform
    C = wavedec2(x,4,'db4');

    % add the abs value of every wavelet coefficient
    cost = norm(C(:),1);

end