close all;clear all;clc
[~,txt,~] = xlsread('text.xlsx');

fpf123=[];
% 
ppmax = 2055;
for i = 1:length(txt)
    filename1=['C:\Users\yy\Desktop\yydata\MyDatabase\original\',txt{i}];
    filename2=['C:\Users\yy\Desktop\yydata\MyDatabase\edit_cut\',txt{i}];
    filename3=['C:\Users\yy\Desktop\yydata\MyDatabase\edit_insert\',txt{i}];
    
    
    [F1o,F2o,po]=gf_50Hz(filename1,10,2000);
    [F3o,fo]=Ex_hilbertIF_F3(filename1,50);

    [F1ec,F2ec,pec]=gf_50Hz(filename2,10,2000);
    [F3ec,fec]=Ex_hilbertIF_F3(filename2,50);
    
    [F1ei,F2ei,pei]=gf_50Hz(filename3,10,2000);
    [F3ei,fei]=Ex_hilbertIF_F3(filename3,50);


    %静态频率特征 194*194
    ffo = ff_pp_194_46(fo,194);  
    ffec = ff_pp_194_46(fec,194);   
    ffei = ff_pp_194_46(fei,194);
    %时序相位表征 

    [fram_num,fram_len,po1_res] = anew_fram_encode(po,25,"next","fram_len",ppmax);
    [~,~,pe11_res] = anew_fram_encode(pec,25,"next","fram_len",ppmax);
    [~,~,pe12_res] = anew_fram_encode(pei,25,"next","fram_len",ppmax);
    i
    fram_num
    fram_len
    fpf123=[fpf123;ffo,po1_res,F1o,F2o,F3o,1;ffec,pe11_res,F1ec,F2ec,F3ec,0;ffei,pe12_res,F1ei,F2ei,F3ei,0];

end

dlmwrite('f_194_p_2583_f123_753.txt', fpf123)
