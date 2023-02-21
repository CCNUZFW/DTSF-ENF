close all;clear all;clc
[~,txto,~] = xlsread('base_12_or_list_150.xlsx');
[~,txte,~] = xlsread('base_12_ed_list_150.xlsx');
fpf123=[];
% 
ppmax = 2055;

for i = 1:length(txto)
    filename1=[txto{i}];
    filename2=[txte{i}];
    
    [F1o,F2o,po]=gf_60Hz(filename1,10,2000);
    [F3o,fo]=Ex_hilbertIF_F3(filename1,60);

    [F1e,F2e,pe]=gf_60Hz(filename2,10,2000);
    [F3e,fe]=Ex_hilbertIF_F3(filename2,60);
    

    %静态频率特征 194*194
    ffo = ff_pp_194_46(po,46);  
    ffe = ff_pp_194_46(pe,46);   

    %时序相位表征 
    [fram_num,fram_len,po1_res] = anew_fram_encode(po,25,"next","fram_len",ppmax);
    [~,~,pe1_res] = anew_fram_encode(pe,25,"next","fram_len",ppmax);

    i
    fram_num
    fram_len
    fpf123=[fpf123;ffo,po1_res,F1o,F2o,F3o,1;ffe,pe1_res,F1e,F2e,F3e,0];
%    


end

%
[~,txteo,~] = xlsread('base_e_or_100.xlsx');
[~,txtee,~] = xlsread('base_e_ed_100.xlsx');
for j = 1:length(txteo)
    filename1=[txteo{j}];
    filename2=[txtee{j}];
    
    [F1o,F2o,po]=gf_50Hz(filename1,10,2000);
    [F3o,fo]=Ex_hilbertIF_F3(filename1,50);

    [F1e,F2e,pe]=gf_50Hz(filename2,10,2000);
    [F3e,fe]=Ex_hilbertIF_F3(filename2,50);
    
    %静态频率特征 194*194
    ffo = ff_pp_194_46(po,46);  
    ffe = ff_pp_194_46(pe,46);  

    %时序相位表征 
    [fram_num,fram_len,po1_res] = anew_fram_encode(po,25,"next","fram_len",ppmax);
    [~,~,pe1_res] = anew_fram_encode(pe,25,"next","fram_len",ppmax);

    j
    fram_num
    fram_len
    fpf123=[fpf123;ffo,po1_res,F1o,F2o,F3o,1;ffe,pe1_res,F1e,F2e,F3e,0];
end

dlmwrite('p46_p2583_f123_500.txt', fpf123)

