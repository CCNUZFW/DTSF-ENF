function [ F3,f] = Ex_hilbertIF_F3(filen,ENF )

% Loads filter coefficients  60HZ滤波器
% load filter_60Hz_1p4Hz_4th  % loads filter coeff. a and b (elliptic bandpass centered at 60 Hz)
load filter_PB20_20Hz_5th % b20 a20 (elliptic lowpass filter with 20Hz passband and zero at 60 Hz)

%%50HZ带通滤波器
if(ENF==50)
    fp2 = 46;fp = 49.5;fs = 50.5;fs2 = 54;  %%47.2  49.7  50.3  52.8 波纹0.5 这个设置还可以  
    Fs = 1000;Fs2 = Fs/2;
    Wp = [fp/Fs2,fs/Fs2];Ws = [fp2/Fs2,fs2/Fs2];
    Rp =1 ;Rs =100 ;  %通带涟漪和阻带衰减
    
   [n,Wn]=ellipord(Wp,Ws,Rp,Rs);
   [b,a]=ellip(n,Rp,Rs,Wn);
   fsd = 1000;
else
    load filter_60Hz_1p4Hz_4th 
    fsd = 1200;
end


% p1=150;
% p2=60; % minimum duration of a utterance (in ms)
br=2000; % number of samples to be discarded in the beginning and end of the detection signals (due to trasnsients) 


[x,fs]=audioread(filen);   % reads wavfile
%x=x(1:12*fs);
x=x./(.81*max(abs(x)));  
%fsd=1200;   % new sampling rate


xds=resample(x,fsd,fs);
%xds=resample(x,100,3675,20); % downsampling from 44.1 Hz to 1200 Hz  
%clear x vad
%% 活动语音检测
% [vad_xds,ind0d,ind1d]=vad_v3(xds,fsd,p1,p2);
% vad_xds=vad_xds(br:end-br-1);
%%


xds_f=filtfilt(b,a,xds); % bandpass filters (around 60Hz) the downsampled signal 
%--------------------使用希尔伯特变换进行瞬时频率估计----------------------------------
z=hilbert(xds_f);  % calculates the analytic signal associated with xds_f
f=phase(z(2:end,:).*conj(z(1:end-1,:)));  % Hilbert estimate of the instantaneous frequency of xds_f  
%clear z xds xds_f  
%-------------------------------------------------------------------------------------
xds=xds(br:end-br-1);

xds_f=xds_f(br:end-br-1);

f=f-median(f(br:end-br));  % removes median value from the instananeous frequency of xds_f 
% Here we are interested in the variations of f 
f=filtfilt(b20,a20,f); % lowpass filters f (20Hz passband) to remove the 60 Hz oscillations that leaks into the Hilbert estimate
f=f(br:end-br);
f=f-mean(f);


derivada1=f(2:end)-f(1:end-1);


aux_der         =derivada1;
aux_der         =aux_der-mean(aux_der);
aux_der         =aux_der.*aux_der;
F3         =sum(abs(aux_der))/(length(aux_der));
F3         =100*log10(F3);
f = abs(f);
% tb=(0:length(xds)-1)/fsd;
% plot(tb,f,'linewidth',2);

end

