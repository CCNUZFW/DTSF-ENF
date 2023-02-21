function [ff]=ff_pp_194_46(x,chang)


fe=x;
len_sig = length(fe);
a = ceil((len_sig-chang)/(chang-1));
% ¶Ôsig½øÐÐpad
b = a*(chang-1)+chang;
fe = [fe;zeros(b-len_sig,1)];

all_duan=[1,chang];
for i=1:(chang-1)
    st = all_duan(i,2)-(chang-a);
    en = st+chang;
    d = [st,en];
    all_duan=[all_duan;d];
end

first_fe=fe(all_duan(1,1):all_duan(1,2));
fea=[first_fe];
for j=2:chang
    ind_s=all_duan(j,1);
    ind_e=all_duan(j,2);
    zj=fe(ind_s:ind_e-1);
    fea=[fea;zj];
end
ff=fea';
end

