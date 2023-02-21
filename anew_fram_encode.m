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

%���ȸ���fram_num_or_len,�� sig_len_max �����fram_len fram_num
if fram_num_or_len=="fram_len"
    fram_len = fram_parameter;
    fram_num = ceil(sig_len_max/fram_len);
elseif fram_num_or_len=="fram_num"
    fram_num = fram_parameter;
    fram_len = ceil(sig_len_max/fram_num);
end

len_sig = length(signal);
% �Ѿ������ fram_len ��fram_num  
% ��ȡδ����pad �ľ���

% �ȼ����ÿ֡�����ٸ��㣬 ��floor ����ĵ�ÿ֡��һ�� 
% ÿ֡���ٸ�����
per_fram_sample_num = floor(len_sig/fram_num);
%ǰ��remain_sample_num��frame�Ⱥ����fram_num-remain_sample_num֡Ҫ��һ����
remain_sample_num = len_sig-per_fram_sample_num*fram_num;

% ��ȡһ��δ����pad�ľ��� ����shape =��per_fram_sample_num+1��fram_num��


%���ܳ���remain_sample_num==0�����
if remain_sample_num==0
% ��==0ʱ ÿ֡�պ�perfram���㣬��ȡ����shape=(per_fram_sample_num,fram_num)
    signal_need = signal'; %ת��
    no_pad_matrix = [];
    for i=1:fram_num
        row_sample = signal_need((i-1)*per_fram_sample_num+1:i*per_fram_sample_num);
        no_pad_matrix=[no_pad_matrix;row_sample];
    end
elseif remain_sample_num~=0
% ��Ҫ����ƴ�� ���Ȼ�ȡǰremain_sample_num ֡ shape=(remain_sample_num��fram_num)
    signal_need_1 = signal(1:(per_fram_sample_num+1)*remain_sample_num)'; %ת��
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

% ���ڵõ��� no_pad_matrix �� no_pad_matrix_1��no_pad_matrix_2
% next ��֡
% same pad ���һ��ֵ
if mode=="next"
    % ==0 ���� no_pad_matrix ,~= ����no_pad_matrix_1��no_pad_matrix_2
    if remain_sample_num==0
        % ���һ��pad same  Ȼ��Ӻ���ǰ����������һ��ǰ�沿�ּ��뵽ǰһ�У�ֱ��fram_len
        last_fram = no_pad_matrix(fram_num,:);
        last_fram_pad_sample = last_fram(per_fram_sample_num); %����������samepad
        
        for i=1:(fram_len-per_fram_sample_num)
            last_fram=[last_fram,last_fram_pad_sample];
        end
        
        res =[];
        next_pad_row = last_fram;
        for i=1:fram_num
            % ��������pad
            wait_pad=no_pad_matrix(fram_num-i+1,:);
            
            que_shao_sample_num = fram_len-length(wait_pad);
            
            wait_pad=[wait_pad,next_pad_row(1:que_shao_sample_num)];
            res=[wait_pad;res];
            %����nextpadrow
            next_pad_row=wait_pad;
        end

    elseif remain_sample_num~=0
        
        % ��pad nopad_matrix_2
        % ���һ��pad same  Ȼ��Ӻ���ǰ����������һ��ǰ�沿�ּ��뵽ǰһ�У�ֱ��fram_len
        
        last_fram = no_pad_matrix_2(fram_num-remain_sample_num,:);
        last_fram_pad_sample = last_fram(per_fram_sample_num); %����������samepad
        
        
        for i=1:(fram_len-per_fram_sample_num)
            last_fram=[last_fram,last_fram_pad_sample];
        end
        
        res_2 =[];
        next_pad_row = last_fram;
        
        matrix_2_fram_num = fram_num-remain_sample_num;
        for i=1:matrix_2_fram_num
            % ��������pad
            wait_pad=no_pad_matrix_2(matrix_2_fram_num-i+1,:);           
            que_shao_sample_num = fram_len-length(wait_pad);      
            wait_pad=[wait_pad,next_pad_row(1:que_shao_sample_num)];
            res_2=[wait_pad;res_2];
            %����nextpadrow
            next_pad_row=wait_pad;
        end
        
        next_pad_row = res_2(1,:);
        res_1=[];
        for i=1:remain_sample_num
            wait_pad=no_pad_matrix_1(remain_sample_num-i+1,:); 
            que_shao_sample_num = fram_len-length(wait_pad);      
            wait_pad=[wait_pad,next_pad_row(1:que_shao_sample_num)];
            res_1=[wait_pad;res_1];
            %����nextpadrow
            next_pad_row=wait_pad;
        end
        res = [res_1;res_2];
    end

elseif mode=="same"
    % ֱ�ӱ���ÿһ�� �Ѻ���pad
    if remain_sample_num==0
        res = [];
        for i=1:fram_num
            % ��������pad
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
            % ��������pad
            wait_pad=no_pad_matrix_1(i,:);
            pad_sample = wait_pad(per_fram_sample_num+1);  
            for j=1:(fram_len-per_fram_sample_num-1)
                wait_pad=[wait_pad,pad_sample];
            end
            res_1=[res_1;wait_pad];
        end
        res_2=[];
        for i=1:(fram_num-remain_sample_num)
            % ��������pad
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
%ת�ú���չ����python���ǰ���
%python��reshape���ԭ���� ת�ñ��һ��
final_res=final_res(:)';
end