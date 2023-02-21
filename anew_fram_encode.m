function [fram_num,fram_len,final_res] = anew_fram_encode(signal,fram_parameter,pad_mode,fram_mode,max_len)




% fram_num fram_len
% fram_len = 10;
% fram_num_or_len = "fram_len";

% freq_len_max = 37281;
% phase_len_max = 2055;
% mode = "next";
% mode = "same";
% signal = po;

sig_len_max = max_len;

mode = pad_mode;
fram_num_or_len = fram_mode;

%首先根据fram_num_or_len,与 sig_len_max 计算出fram_len fram_num
if fram_num_or_len=="fram_len"
    fram_len = fram_parameter;
    fram_num = ceil(sig_len_max/fram_len);
elseif fram_num_or_len=="fram_num"
    fram_num = fram_parameter;
    fram_len = ceil(sig_len_max/fram_num);
end

len_sig = length(signal);
% 已经获得了 fram_len 与fram_num  
% 获取未经过pad 的矩阵

% 先计算出每帧共多少个点， 用floor 多余的点每帧给一个 
% 每帧多少个样点
per_fram_sample_num = floor(len_sig/fram_num);
%前面remain_sample_num个frame比后面的fram_num-remain_sample_num帧要多一个点
remain_sample_num = len_sig-per_fram_sample_num*fram_num;

% 获取一个未经过pad的矩阵 矩阵shape =（per_fram_sample_num+1，fram_num）


%可能出现remain_sample_num==0的情况
if remain_sample_num==0
% 当==0时 每帧刚好perfram个点，获取矩阵shape=(per_fram_sample_num,fram_num)
    signal_need = signal'; %转置
    no_pad_matrix = [];
    for i=1:fram_num
        row_sample = signal_need((i-1)*per_fram_sample_num+1:i*per_fram_sample_num);
        no_pad_matrix=[no_pad_matrix;row_sample];
    end
elseif remain_sample_num~=0
% 需要进行拼接 ，先获取前remain_sample_num 帧 shape=(remain_sample_num，fram_num)
    signal_need_1 = signal(1:(per_fram_sample_num+1)*remain_sample_num)'; %转置
    no_pad_matrix_1=[];
    for i=1:remain_sample_num
        row_sample = signal_need_1((i-1)*(per_fram_sample_num+1)+1:i*(per_fram_sample_num+1));
        no_pad_matrix_1=[no_pad_matrix_1;row_sample];
    end
    
    signal_need_2 = signal((per_fram_sample_num+1)*remain_sample_num+1:len_sig)';
    no_pad_matrix_2=[];
    for i=1:(fram_num-remain_sample_num)
        
        row_sample = signal_need_2((i-1)*per_fram_sample_num+1:i*per_fram_sample_num);
        no_pad_matrix_2=[no_pad_matrix_2;row_sample];
    end
end

% 现在得到了 no_pad_matrix 或 no_pad_matrix_1和no_pad_matrix_2
% next 分帧
% same pad 最后一个值
if mode=="next"
    % ==0 则获得 no_pad_matrix ,~= 则是no_pad_matrix_1和no_pad_matrix_2
    if remain_sample_num==0
        % 最后一行pad same  然后从后向前遍历，将后一行前面部分加入到前一行，直到fram_len
        last_fram = no_pad_matrix(fram_num,:);
        last_fram_pad_sample = last_fram(per_fram_sample_num); %用这个点进行samepad
        
        for i=1:(fram_len-per_fram_sample_num)
            last_fram=[last_fram,last_fram_pad_sample];
        end
        
        res =[];
        next_pad_row = last_fram;
        for i=1:fram_num
            % 即将进行pad
            wait_pad=no_pad_matrix(fram_num-i+1,:);
            
            que_shao_sample_num = fram_len-length(wait_pad);
            
            wait_pad=[wait_pad,next_pad_row(1:que_shao_sample_num)];
            res=[wait_pad;res];
            %更新nextpadrow
            next_pad_row=wait_pad;
        end

    elseif remain_sample_num~=0
        
        % 先pad nopad_matrix_2
        % 最后一行pad same  然后从后向前遍历，将后一行前面部分加入到前一行，直到fram_len
        
        last_fram = no_pad_matrix_2(fram_num-remain_sample_num,:);
        last_fram_pad_sample = last_fram(per_fram_sample_num); %用这个点进行samepad
        
        
        for i=1:(fram_len-per_fram_sample_num)
            last_fram=[last_fram,last_fram_pad_sample];
        end
        
        res_2 =[];
        next_pad_row = last_fram;
        
        matrix_2_fram_num = fram_num-remain_sample_num;
        for i=1:matrix_2_fram_num
            % 即将进行pad
            wait_pad=no_pad_matrix_2(matrix_2_fram_num-i+1,:);           
            que_shao_sample_num = fram_len-length(wait_pad);      
            wait_pad=[wait_pad,next_pad_row(1:que_shao_sample_num)];
            res_2=[wait_pad;res_2];
            %更新nextpadrow
            next_pad_row=wait_pad;
        end
        
        next_pad_row = res_2(1,:);
        res_1=[];
        for i=1:remain_sample_num
            wait_pad=no_pad_matrix_1(remain_sample_num-i+1,:); 
            que_shao_sample_num = fram_len-length(wait_pad);      
            wait_pad=[wait_pad,next_pad_row(1:que_shao_sample_num)];
            res_1=[wait_pad;res_1];
            %更新nextpadrow
            next_pad_row=wait_pad;
        end
        res = [res_1;res_2];
    end

elseif mode=="same"
    % 直接遍历每一行 把后面pad
    if remain_sample_num==0
        res = [];
        for i=1:fram_num
            % 即将进行pad
            wait_pad=no_pad_matrix(i,:);
            pad_sample = wait_pad(per_fram_sample_num);  
            for j=1:(fram_len-per_fram_sample_num)
                wait_pad=[wait_pad,pad_sample];
            end
            res=[res;wait_pad];
        end

    elseif remain_sample_num~=0
        res_1=[];
        for i=1:remain_sample_num
            % 即将进行pad
            wait_pad=no_pad_matrix_1(i,:);
            pad_sample = wait_pad(per_fram_sample_num+1);  
            for j=1:(fram_len-per_fram_sample_num-1)
                wait_pad=[wait_pad,pad_sample];
            end
            res_1=[res_1;wait_pad];
        end
        res_2=[];
        for i=1:(fram_num-remain_sample_num)
            % 即将进行pad
            wait_pad=no_pad_matrix_2(i,:);
            pad_sample = wait_pad(per_fram_sample_num);  
            for j=1:(fram_len-per_fram_sample_num)
                wait_pad=[wait_pad,pad_sample];
            end
            res_2=[res_2;wait_pad];
        end
        res = [res_1;res_2];
    end
end

final_res = res';
%转置后按列展开，python中是按行
%python中reshape获得原矩阵 转置变成一行
final_res=final_res(:)';
end