function cost = cost_l1_wav3D(x)

    % take the wavelet transform
    wd = wavedec3(x,4,'db4');

    % add the abs value of every wavelet coefficient
    cost = 0;
    for i = 1:length(wd.dec)
        coeffs_i = wd.dec{i};
        cost = cost + norm(coeffs_i(:),1);
    end

end