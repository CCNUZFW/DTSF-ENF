function [F1,F2,phase2]=gf_60Hz(arquivo,num_periodos,Nfft)
%function [F2,phase2]=grafico_fase_60Hz(arquivo,num_periodos,Nfft)
%vari�veis de saida  �������
%F1 --> feauture usando DFT  DFT����
%F2 --> feature usando DFT^1  DFT1����

%vari�veis de entrada  �������
%arquivo --> localiza��o da grava��o (pasta, diret�rio, subdiret�rio, etc)
%deltaf--> largura de faixa do filtro [Hz]    
%num_periodos --> tamanho da janela (n�mero de ciclos da ENF nominal)
%���ڴ�С����ENF������������
%Nfft --> n�mero de pontos da DFT  DFT�ĵ���

[samples,fs_original]=audioread(arquivo);
tam= length(samples);    %ԭʼ�źų���

fs=1200;
fc=60;

original=samples;
original=original-mean(original);    %ȥ��ֵ����

%Para fazer o sub-amostragem a 1200Hz --> diminuir a carga de processamento
%computacional
%��1200HZ�²��������ͼ��㸺��
original_f2=resample(original,fs,fs_original);
tam2=length(original_f2);   %�²���֮��ĵ��źų���

%Para o filtro passa baixas agudo  
%���д�ͨ�˲�

deltaf = 0.6;
f1=fc-deltaf/2;
f2=fc+deltaf/2;

%para normalizar os valores: [0 fs/2]--> [0 pi]  �ı�Ƶ�ʱ仯�߶�
w1=f1*2*pi/fs; 
w2=f2*2*pi/fs;

%para normalizar os valores: [0 pi] --> [0 1]    Ƶ�ʹ�һ����Wn��ϵ��������[0 1]֮�����
Wn=[w1 w2]/pi;  

%N�mero de coeficientes do filtro
%FIR�����˲���������FIR�˲����Ĵ��������������������ʹ������1������=10001
N=10000; 

%Requerimento de filtfilt   �˲������
if tam2<=3*N,    %�������С���������˲�������
    h=fir1(N,Wn,'bandpass'); %Para definir a resposta impulsiva do filtro   �˲�����������Ӧ��ȷ�� �Ӵ�������λ�˲���
    original_f2=[zeros(1.5*N-round(tam2/2)+50,1);original_f2;zeros(1.5*N+50-round(tam2/2),1)];  %�ź�ǰ�����һ������㣬ȫ����Ϊ3*N+100
    orig_filtrada=filtfilt(h,1,original_f2); %Para executar um filtro sem defazamento   �˲����˲�,ʹ��filtfilt������ʱ
    original_f2=original_f2(1.5*N-round(tam2/2)+50+1:1.5*N-round(tam2/2)+50+tam2);   %Eliminar o excesso de zeros  ȥ�����
    orig_filtrada=orig_filtrada(1.5*N-round(tam2/2)+50+1:1.5*N-round(tam2/2)+50+tam2); %Eliminar o excesso de zeros �˲�����ź�ȥ�����
else
    h=fir1(N,Wn,'bandpass'); %Para definir a resposta impulsiva do filtro
    original_f2=[zeros(1000,1);original_f2;zeros(1000,1)];       % ǰ�������1000����
    orig_filtrada=filtfilt(h,1,original_f2); %Para executar um filtro sem defazamento
    original_f2=original_f2(1001:1000+tam2);   %Eliminar o excesso de zeros 
    orig_filtrada=orig_filtrada(1001:1000+tam2); %Eliminar o excesso de zeros    �õ��˲�����ź�
    
end




%Para o processo de seguimento de fase na ENF (60 Hz em Espanha)
%orig_filtrada=resample(orig_filtrada,10000,fs);
tam2=length(orig_filtrada);

%Para o processo de seguimento de fase na ENF (50 Hz em Espanha)
t_periodo=1/60;                   %һ��ENF������������ʱ��
amostras_periodo=t_periodo*fs;    %һ��ENF�������ж��ٸ�������

%num_periodos=3;
%num_periodos=10;
t_bloco=num_periodos/60;          %�������num_periodos����һ������ʱ�����õ���֡������ʱ�䵥λ��
amostras_bloco=t_bloco*fs;        %������֡���������������

%�źŷ�֡�õ�֡����֡��Ϊnum_periodos*amostras_periodo��ͬamostras_bloco����֡��Ϊamostras_periodo���õ�֡��
n_blocos=fix(tam2/amostras_periodo)-(num_periodos-1);   

inicio=1:amostras_periodo:(n_blocos-1)*amostras_periodo+1;  %��֡����źţ�ÿ֡�źŵ�һ���������

fim=inicio+amostras_bloco-1;                                %��֡����źţ�ÿ֡�ź����һ���������

f1                =  zeros(n_blocos,1);                     %DFT����Ƶ�ʣ���֡�������ο��� 
f2                =  f1;                                    %DFT1����Ƶ��
phase1            =  f1;                                    %DFT������λ
phase2            =  f1;                                    %DFT1������λ
der_orig_filtrada=fs*diff([0; orig_filtrada]);              %��ʼDFT1�ļ�����̣�1��

NFFT_2=Nfft/2;
for i=1:n_blocos,     %��֡��ѭ��
    
%     if mod(i,200)==0  %��ǰ֡����200��ģ
%        clc;
%        disp('Processando ...'); %����ַ���
%        disp (i);                %���i��ֵ��i��С��ż��
%        disp('out of'); 
%        disp(n_blocos)           %��֡��
%     end
    %[fftaux1,W]=freqz(hanning(amostras_bloco).*orig_filtrada(inicio(i):fim(i)),1,5000);
    naco    =     orig_filtrada(inicio(i):fim(i));       %�õ�ԭʼ�źŵķ�֡�ź�
    nacoDOT =     der_orig_filtrada(inicio(i):fim(i));   %�õ�һ�ײ���źŵķ�֡�ź�
    janela=hanning(amostras_bloco);
    xj=janela.*naco;                      %һ֡DFT�źų��Դ�����
    [XJ,W]=freqz(xj,1,NFFT_2);            %��DFTÿ֡�ź�Ƶ����Ӧ
    
    %  % (1) frame sem janelamento:
    % naco=[x(k); naco(1:(windowsize-1))];
    % % (2) DFT^0 janelado:
    % xj=janela.*naco;
    % [XJ,W]=freqz(xj,1,10000);
    % % (3) obtendo xDOT:
    % nacoDOT=[xdot(k); nacoDOT(1:(windowsize-1))];
    % (4) janelar xDOT:
    xjDOT=janela.*nacoDOT;            %һ֡DFT1�źų��Դ�����
    % (5) DFT^1 = DFT{xDOT janelado}:
    [XJDOT,W]=freqz(xjDOT,1,NFFT_2);  %��DFT1ÿ֡�ź�Ƶ����Ӧ
    % (6) DFT1 = DFT1*F(w):
    XJDOT=XJDOT.*W./(2*sin(W/2));     %���DFT[K]��F(K)�������ϵĲ�һ��������
    % (7) Achar Kmax e computar f(t):
    [maxXJ,Kmax]=max(abs(XJ));
    f1(i)=(fs/2)*W(Kmax)/pi;  %(fs/2)*W(101)/pi = 50Hz  DFT����Ƶ�ʣ�HZ��λ
    f2(i)=abs(XJDOT(Kmax))/(2*pi*abs(XJ(Kmax)));        %DFT1����Ƶ�ʣ��Ƕȵ�λ
    
    phase1(i)=angle(XJ(Kmax));                 %���DFT�任�õ�����λ
    cw0=cos(2*pi*f2(i)/fs);
    sw0=sin(2*pi*f2(i)/fs);
    %phase2(k)=180*(atan((tan(angle(XJDOT(Kmax)))*(1-cw0)-sw0)/(1-cw0+tan(angle(XJDOT(Kmax)))*sw0)))/pi; 
    %phase2(k)=180*(atan((tan(angle(XJDOT(Kmax))-pi)*(1-cw0)-sw0)/(1-cw0+tan(angle(XJDOT(Kmax))-pi)*sw0)))/pi;
    %phase2(i)=(atan((tan(angle(XJDOT(Kmax)))*(1-cw0)-sw0)/(1-cw0+tan(angle(XJDOT(Kmax)))*sw0)));
    %���Բ�ֵ������ƽ����
    ind_men_k_freq=floor(f2(i)*Nfft/fs)+1;     %floor����ȡ�������˸�һ����������
    ind_maj_k_freq=ind_men_k_freq+1;           %����ȡ��
    
    f_men=(fs/2)*W(ind_men_k_freq)/pi;         %KlowƵ��
    f_maj=(fs/2)*W(ind_maj_k_freq)/pi;         %KhighƵ��
    
    phase_men=(angle(XJDOT(ind_men_k_freq))); %/pi*180 %para theta Klow��λ
    phase_maj=(angle(XJDOT(ind_maj_k_freq))); %/pi*180 %para theta Khigh��λ
    
    phase_aux=[phase_men phase_maj];
    phase_aux=unwrap(phase_aux);              %�����
    
    phase_men=phase_aux(1);
    phase_maj=phase_aux(2);
    
    phase_theta=(phase_maj-phase_men)*(f2(i)-f_men)/(f_maj-f_men)+phase_men;  %���������ֲ�һ��
    
    phase2(i)=atan( ( tan(phase_theta)*(1-cw0)-sw0 )/( 1 - cw0 + tan(phase_theta)*sw0 ) );  %���DFT1������λֵ
        
end

%===========================================================
%valida��o de phase2  CH��phase2 ����
%============================================================

tam_phase=length(f1);
aux_phase2=zeros(tam_phase,2);

ref_phase=[phase1 phase1];                   %�ο���λ
for i=1:tam_phase,
    
    if phase2(i)<0,
        aux_phase2(i,1)= pi+phase2(i);       %С��0��ֵ���ܳ�����[pi/2,pi]������
        aux_phase2(i,2)= phase2(i);
    else
        aux_phase2(i,1)= phase2(i);
        aux_phase2(i,2)= -(pi-phase2(i));    %����0��ֵ���ܳ�����[-pi,-pi/2]������
    end
end
 
distan_phase=aux_phase2-ref_phase;           %��ο���λ�Ĳ�
distan_phase=valida_phase(distan_phase);

distan_phase=abs(distan_phase);
[val, indices_phase]=min(distan_phase');     %
indices_phase=indices_phase';                %ȡ�����С����λ������ֵ

for i=1:tam_phase,
    
    phase2(i)=aux_phase2(i,indices_phase(i)); %��λ����
end

%===========================================================


phase1=unwrap(phase1);     %�����
                     
phase1=180*phase1/pi;      %������ֵ��Ϊ�Ƕ�ֵ


phase2=unwrap(phase2);

phase2=180*phase2/pi;

%================================================
%Para o valor da derivada
%���ź������
derivada1=phase1(2:end)-phase1(1:end-1);   
derivada2=phase2(2:end)-phase2(1:end-1);




%====================================================================
%Para as medidas de avalia��o CH:������ۣ���ȡ����F
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

