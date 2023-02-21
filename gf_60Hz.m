function [F1,F2,phase2]=gf_60Hz(arquivo,num_periodos,Nfft)
%function [F2,phase2]=grafico_fase_60Hz(arquivo,num_periodos,Nfft)
%vari醰eis de saida  输出参数
%F1 --> feauture usando DFT  DFT特征
%F2 --> feature usando DFT^1  DFT1特征

%vari醰eis de entrada  输入参数
%arquivo --> localiza玢o da grava玢o (pasta, diret髍io, subdiret髍io, etc)
%deltaf--> largura de faixa do filtro [Hz]    
%num_periodos --> tamanho da janela (n鷐ero de ciclos da ENF nominal)
%窗口大小，以ENF的周期数计算
%Nfft --> n鷐ero de pontos da DFT  DFT的点数

[samples,fs_original]=audioread(arquivo);
tam= length(samples);    %原始信号长度

fs=1200;
fc=60;

original=samples;
original=original-mean(original);    %去中值分量

%Para fazer o sub-amostragem a 1200Hz --> diminuir a carga de processamento
%computacional
%以1200HZ下采样，降低计算负荷
original_f2=resample(original,fs,fs_original);
tam2=length(original_f2);   %下采样之后的的信号长度

%Para o filtro passa baixas agudo  
%进行带通滤波

deltaf = 0.6;
f1=fc-deltaf/2;
f2=fc+deltaf/2;

%para normalizar os valores: [0 fs/2]--> [0 pi]  改变频率变化尺度
w1=f1*2*pi/fs; 
w2=f2*2*pi/fs;

%para normalizar os valores: [0 pi] --> [0 1]    频率归一化，Wn的系数必须是[0 1]之间的数
Wn=[w1 w2]/pi;  

%N鷐ero de coeficientes do filtro
%FIR数字滤波器阶数，FIR滤波器的窗长必须是奇数，阶数和窗长相差1，窗长=10001
N=10000; 

%Requerimento de filtfilt   滤波器设计
if tam2<=3*N,    %如果长度小于三倍的滤波器阶数
    h=fir1(N,Wn,'bandpass'); %Para definir a resposta impulsiva do filtro   滤波器的脉冲响应的确定 加窗线性相位滤波器
    original_f2=[zeros(1.5*N-round(tam2/2)+50,1);original_f2;zeros(1.5*N+50-round(tam2/2),1)];  %信号前后加上一定的零点，全长变为3*N+100
    orig_filtrada=filtfilt(h,1,original_f2); %Para executar um filtro sem defazamento   滤波器滤波,使用filtfilt避免延时
    original_f2=original_f2(1.5*N-round(tam2/2)+50+1:1.5*N-round(tam2/2)+50+tam2);   %Eliminar o excesso de zeros  去除零点
    orig_filtrada=orig_filtrada(1.5*N-round(tam2/2)+50+1:1.5*N-round(tam2/2)+50+tam2); %Eliminar o excesso de zeros 滤波后的信号去除零点
else
    h=fir1(N,Wn,'bandpass'); %Para definir a resposta impulsiva do filtro
    original_f2=[zeros(1000,1);original_f2;zeros(1000,1)];       % 前后各加上1000个点
    orig_filtrada=filtfilt(h,1,original_f2); %Para executar um filtro sem defazamento
    original_f2=original_f2(1001:1000+tam2);   %Eliminar o excesso de zeros 
    orig_filtrada=orig_filtrada(1001:1000+tam2); %Eliminar o excesso de zeros    得到滤波后的信号
    
end




%Para o processo de seguimento de fase na ENF (60 Hz em Espanha)
%orig_filtrada=resample(orig_filtrada,10000,fs);
tam2=length(orig_filtrada);

%Para o processo de seguimento de fase na ENF (50 Hz em Espanha)
t_periodo=1/60;                   %一个ENF周期所经历的时长
amostras_periodo=t_periodo*fs;    %一个ENF周期内有多少个采样点

%num_periodos=3;
%num_periodos=10;
t_bloco=num_periodos/60;          %输入参数num_periodos乘以一个周期时长，得到分帧窗长（时间单位）
amostras_bloco=t_bloco*fs;        %窗长或帧长（采样点个数）

%信号分帧得到帧数，帧长为num_periodos*amostras_periodo（同amostras_bloco），帧移为amostras_periodo，得到帧数
n_blocos=fix(tam2/amostras_periodo)-(num_periodos-1);   

inicio=1:amostras_periodo:(n_blocos-1)*amostras_periodo+1;  %分帧后的信号，每帧信号第一个点的索引

fim=inicio+amostras_bloco-1;                                %分帧后的信号，每帧信号最后一个点的索引

f1                =  zeros(n_blocos,1);                     %DFT估计频率，以帧数建立参考点 
f2                =  f1;                                    %DFT1估计频率
phase1            =  f1;                                    %DFT估计相位
phase2            =  f1;                                    %DFT1估计相位
der_orig_filtrada=fs*diff([0; orig_filtrada]);              %开始DFT1的计算过程（1）

NFFT_2=Nfft/2;
for i=1:n_blocos,     %以帧数循环
    
%     if mod(i,200)==0  %当前帧除以200的模
%        clc;
%        disp('Processando ...'); %输出字符串
%        disp (i);                %输出i的值，i是小于偶数
%        disp('out of'); 
%        disp(n_blocos)           %总帧数
%     end
    %[fftaux1,W]=freqz(hanning(amostras_bloco).*orig_filtrada(inicio(i):fim(i)),1,5000);
    naco    =     orig_filtrada(inicio(i):fim(i));       %得到原始信号的分帧信号
    nacoDOT =     der_orig_filtrada(inicio(i):fim(i));   %得到一阶差分信号的分帧信号
    janela=hanning(amostras_bloco);
    xj=janela.*naco;                      %一帧DFT信号乘以窗函数
    [XJ,W]=freqz(xj,1,NFFT_2);            %求DFT每帧信号频率响应
    
    %  % (1) frame sem janelamento:
    % naco=[x(k); naco(1:(windowsize-1))];
    % % (2) DFT^0 janelado:
    % xj=janela.*naco;
    % [XJ,W]=freqz(xj,1,10000);
    % % (3) obtendo xDOT:
    % nacoDOT=[xdot(k); nacoDOT(1:(windowsize-1))];
    % (4) janelar xDOT:
    xjDOT=janela.*nacoDOT;            %一帧DFT1信号乘以窗函数
    % (5) DFT^1 = DFT{xDOT janelado}:
    [XJDOT,W]=freqz(xjDOT,1,NFFT_2);  %求DFT1每帧信号频率响应
    % (6) DFT1 = DFT1*F(w):
    XJDOT=XJDOT.*W./(2*sin(W/2));     %求解DFT[K]，F(K)和论文上的不一样？？？
    % (7) Achar Kmax e computar f(t):
    [maxXJ,Kmax]=max(abs(XJ));
    f1(i)=(fs/2)*W(Kmax)/pi;  %(fs/2)*W(101)/pi = 50Hz  DFT估计频率，HZ单位
    f2(i)=abs(XJDOT(Kmax))/(2*pi*abs(XJ(Kmax)));        %DFT1估计频率，角度单位
    
    phase1(i)=angle(XJ(Kmax));                 %求得DFT变换得到的相位
    cw0=cos(2*pi*f2(i)/fs);
    sw0=sin(2*pi*f2(i)/fs);
    %phase2(k)=180*(atan((tan(angle(XJDOT(Kmax)))*(1-cw0)-sw0)/(1-cw0+tan(angle(XJDOT(Kmax)))*sw0)))/pi; 
    %phase2(k)=180*(atan((tan(angle(XJDOT(Kmax))-pi)*(1-cw0)-sw0)/(1-cw0+tan(angle(XJDOT(Kmax))-pi)*sw0)))/pi;
    %phase2(i)=(atan((tan(angle(XJDOT(Kmax)))*(1-cw0)-sw0)/(1-cw0+tan(angle(XJDOT(Kmax)))*sw0)));
    %线性插值，增加平滑度
    ind_men_k_freq=floor(f2(i)*Nfft/fs)+1;     %floor向下取整，加了个一？？？错了
    ind_maj_k_freq=ind_men_k_freq+1;           %向上取整
    
    f_men=(fs/2)*W(ind_men_k_freq)/pi;         %Klow频率
    f_maj=(fs/2)*W(ind_maj_k_freq)/pi;         %Khigh频率
    
    phase_men=(angle(XJDOT(ind_men_k_freq))); %/pi*180 %para theta Klow相位
    phase_maj=(angle(XJDOT(ind_maj_k_freq))); %/pi*180 %para theta Khigh相位
    
    phase_aux=[phase_men phase_maj];
    phase_aux=unwrap(phase_aux);              %解卷绕
    
    phase_men=phase_aux(1);
    phase_maj=phase_aux(2);
    
    phase_theta=(phase_maj-phase_men)*(f2(i)-f_men)/(f_maj-f_men)+phase_men;  %和论文上又不一样
    
    phase2(i)=atan( ( tan(phase_theta)*(1-cw0)-sw0 )/( 1 - cw0 + tan(phase_theta)*sw0 ) );  %求得DFT1估计相位值
        
end

%===========================================================
%valida玢o de phase2  CH：phase2 矫正
%============================================================

tam_phase=length(f1);
aux_phase2=zeros(tam_phase,2);

ref_phase=[phase1 phase1];                   %参考相位
for i=1:tam_phase,
    
    if phase2(i)<0,
        aux_phase2(i,1)= pi+phase2(i);       %小于0的值可能出现在[pi/2,pi]区间内
        aux_phase2(i,2)= phase2(i);
    else
        aux_phase2(i,1)= phase2(i);
        aux_phase2(i,2)= -(pi-phase2(i));    %大于0的值可能出现在[-pi,-pi/2]区间内
    end
end
 
distan_phase=aux_phase2-ref_phase;           %与参考相位的差
distan_phase=valida_phase(distan_phase);

distan_phase=abs(distan_phase);
[val, indices_phase]=min(distan_phase');     %
indices_phase=indices_phase';                %取距离更小的相位的索引值

for i=1:tam_phase,
    
    phase2(i)=aux_phase2(i,indices_phase(i)); %相位矫正
end

%===========================================================


phase1=unwrap(phase1);     %解卷绕
                     
phase1=180*phase1/pi;      %将弧度值变为角度值


phase2=unwrap(phase2);

phase2=180*phase2/pi;

%================================================
%Para o valor da derivada
%对信号做差分
derivada1=phase1(2:end)-phase1(1:end-1);   
derivada2=phase2(2:end)-phase2(1:end-1);




%====================================================================
%Para as medidas de avalia玢o CH:结果评价，求取特征F
%====================================================================

aux_der         =derivada1;
aux_der         =aux_der-mean(aux_der);
aux_der         =aux_der.*aux_der;
F1         =sum(abs(aux_der))/(length(aux_der));
F1         =100*log10(F1);                      


aux_der2        =derivada2;
aux_der2        =aux_der2-mean(aux_der2);
aux_der2        =aux_der2.*aux_der2;
F2         =sum(abs(aux_der2))/(length(aux_der2));
F2         =100*log10(F2);

tempo=tam/fs_original;

