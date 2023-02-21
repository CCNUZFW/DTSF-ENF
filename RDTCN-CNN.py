
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


from tensorflow import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
import math
from sklearn.preprocessing import MinMaxScaler
import shutil
import random
# from keras_layer_normalization import LayerNormalization
from keras import layers
from keras.layers import Activation
from keras.layers import add,Conv1D, Dropout
# 内存自增长

# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# for gpu in gpus:
#
#     tf.config.experimental.set_memory_growth(gpu, True)


def get_mode_num_len(data_file):
    filename_no_txt = data_file[0:-4]                # 去除后四位
    str_list_filename = filename_no_txt.split("_")  # 分隔方式_
    pad_mode = str_list_filename[0]
    pad_mode = pad_mode.split("/")
    pad_mode = pad_mode[1]
    fram_num = int(str_list_filename[4])
    fram_len = int(str_list_filename[5])
    print("pad_mode", pad_mode)
    print("fram_num", fram_num)
    print("fram_len", fram_len)
    return pad_mode,fram_num,fram_len


def get_data(data_file):
    data_all = []
    with open(data_file, "r") as f:
        for line in f.readlines():
            data = line.split('\n\t')
            for str in data:
                sub_str = str.split(',')
            if sub_str:
                data_all .append(sub_str)
    f.close()
    data_all = np.array(data_all)
    data_all = data_all.astype(np.float64)
    data_all = data_all
    data_t = data_all[:, 0:-1]
    lable_all = data_all[:, -1]

    lable_all = lable_all
    lable_all = keras.utils.to_categorical(lable_all, num_classes=2)

    print(data_t.shape)
    print(lable_all.shape)
    return data_t, lable_all

def split_data(data_t,fram_num,fram_len):
    phase_len = fram_num*fram_len
    data_0 = data_t[:, 0:phase_len]
    data_1 = data_t[:, phase_len:]
    print("data_0.shape ", data_0.shape)
    print("data_1.shape ", data_1.shape)
    return data_0, data_1



def train_test_split_and_scaler(data,label,fram_num,fram_len):
    x_train_all, x_test, y_train_all, y_test = train_test_split(data, label, test_size=0.1, random_state=17)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train_all, y_train_all, test_size=0.22, random_state=18)
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    x_test = scaler.transform(x_test)

    ###时间
    train_time = x_train.reshape(x_train.shape[0], fram_num, fram_len)
    test_time = x_test.reshape(x_test.shape[0], fram_num, fram_len)
    valid_time = x_valid.reshape(x_valid.shape[0], fram_num, fram_len)
    ###空间
    train_space = x_train.reshape(x_train.shape[0], fram_num, fram_len,1)
    test_space = x_test.reshape(x_test.shape[0], fram_num, fram_len,1)
    valid_space = x_valid.reshape(x_valid.shape[0], fram_num, fram_len,1)


    print(x_train.shape)
    print(x_valid.shape)
    print(x_test.shape)

    return train_time,train_space, y_train, valid_time,valid_space, y_valid, test_time,test_space, y_test

def ResBlock(x,filters,kernel_size,dilation_rate):
    h=Conv1D(filters,kernel_size,padding='same',dilation_rate=dilation_rate,activation='relu')(x) #第一卷积
    s=Conv1D(filters,kernel_size,padding='same',dilation_rate=dilation_rate)(h) #第二卷积
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(s)
    if x.shape[-1]==filters:
        shortcut=x
    else:
        shortcut=Conv1D(filters,kernel_size,padding='same')(x)  #shortcut（捷径）
    o=add([r,s,h,shortcut])
    o=Activation('relu')(o)  #激活函数
    return o

def bulid_model_RDTCN_CNN(learning_rate,fram_num,fram_len):

    input_time = keras.layers.Input(shape=[fram_num, fram_len])
    input_space = keras.layers.Input(shape=[fram_num, fram_len, 1])
# ###时间网络
#     x = keras.layers.Bidirectional(keras.layers.LSTM(fram_len, return_sequences=True, kernel_regularizer='l2'))(
#         input_time)
#     # x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
#     x = Activation('relu')(x)
#
#     x = keras.layers.Bidirectional(keras.layers.LSTM(fram_len, return_sequences=True, kernel_regularizer='l2'))(x)
#     # x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
#     x = Activation('relu')(x)
#     x = keras.layers.Flatten()(x)
#
#     x = keras.layers.Dense(512)(x)
#     x = Activation('relu')(x)
#     x = keras.layers.Dropout(0.2)(x)
#     x_1 = keras.layers.Dense(256)(x)
#     x = Activation('relu')(x_1)
#     output_time = keras.layers.Dense(2, activation='softmax',name='layers_softmax1')(x)

###RDtcn

    hiddenatt2 = keras.layers.Conv2D(16, (3, 3), strides=(3, 3),activation='relu', padding='same', data_format='channels_last', name='att1')(( input_time))
    hiddenatt3 = keras.layers.Conv2D(32, (3, 3), strides=(3, 3),activation='relu', padding='same', data_format='channels_last', name='att2')(hiddenatt2)

    hiddenatt5 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(hiddenatt3)
    # hiddenatt6 = keras.layers.Reshape([160])(hiddenatt5)
    hiddenatt6 = keras.layers.Flatten()(hiddenatt5)
    hiddenatt7 = keras.layers.Dense(fram_num, activation='sigmoid')(hiddenatt6)
    hiddenatt8 = keras.layers.Reshape((fram_num, 1))(hiddenatt7)
    hiddenatt9 = keras.layers.multiply([hiddenatt8, input_time])
    hiddenatt10 = keras.layers.Reshape((fram_num, fram_len))(hiddenatt9)

    # hiddenatt11 = ResBlock(hiddenatt10, filters=6, kernel_size=1, dilation_rate=1)
    # hiddenatt12 = ResBlock(hiddenatt11, filters=16, kernel_size=3, dilation_rate=2)
    hiddenatt13 = ResBlock(hiddenatt10, filters=32, kernel_size=3, dilation_rate=4)

    hiddenatt14 = keras.layers.Flatten()(hiddenatt13)
    hiddenatt15 = keras.layers.Dense(1024, activation='relu', name='layers_fullytcn1')(hiddenatt14)
    hiddenatt16 = keras.layers.Dense(256, activation='relu', name='layers_fullytcn2')(hiddenatt15)
    hiddenatt17 = keras.layers.Dropout(0.2)(hiddenatt16)
    hiddenatt18 = Activation('relu')(hiddenatt17)
    output_time = keras.layers.Dense(2, activation='softmax',name='layers_softmax1')(hiddenatt18)





### 空间网络
    hidden1 = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', data_format='channels_last',
                                  name='layer1_con1')(input_space)
    hidden2 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(hidden1)
    hidden3 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', data_format='channels_last',
                                  name='layer1_con2')(hidden2)
    hidden4 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(hidden3)
    hidden5 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', data_format='channels_last',
                                  name='layer1_con3')(hidden4)
    hidden6 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(hidden5)
    hidden7 = keras.layers.Flatten()(hidden6)
    hidden8 = keras.layers.Dense(1024, activation='relu', name='layers_fully1')(hidden7)
    hidden8_1 = keras.layers.Dense(256, activation='relu', name='layers_fully2')(hidden8)
    hidden9 = keras.layers.Dropout(0.2)(hidden8_1)
    output2_space = keras.layers.Dense(2, activation='softmax', name='layers_softmax2')(hidden9)

###联合
    # hidden17 = keras.layers.concatenate([x_1, hidden8_1])
    hidden17 = keras.layers.concatenate([hiddenatt16, hidden8_1])
    hidden18 = keras.layers.Reshape((2, 256, 1))(hidden17)
    hidden19 = keras.layers.Reshape((2, 256))(hidden18)
    hidden27 = keras.layers.Flatten()(hidden19)
    # hidden28 = keras.layers.Dense(400, activation='relu', name='layers8')(hidden27)

    hidden29 = keras.layers.Dense(256, activation='relu', name='layers11')(hidden27)
    # hidden30 = keras.layers.Dense(128, activation='relu', name='layers12')(hidden29)
    hidden31 = keras.layers.Dense(32, activation='relu', name='layers13')(hidden29)
    hidden32 = keras.layers.Dropout(0.2)(hidden31)
    output3 = keras.layers.Dense(2, activation='softmax', name='fc3')(hidden32)


    model = keras.models.Model(inputs=[input_time,input_space], outputs=[output_time,output2_space,output3])
    model.summary()
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss={
        'layers_softmax1': 'categorical_crossentropy',
        'layers_softmax2': 'categorical_crossentropy',
        'fc3': 'categorical_crossentropy'}, optimizer=optimizer, metrics=['accuracy'])
    return model




def train_save(model,atrain_time,atrain_space, ay_train,avalid_time,avalid_space, ay_valid,epochs, fram_num, fram_len,logdir="logdir0",batch_size=64):
    # model = bulid_model(lr, fram_num, fram_len)  # f

    logdir = logdir  + "_" + str(fram_num) + "_" + str(fram_len)
    if not os.path.exists(logdir):  # 若该文件夹不存在 则创建一个
        os.mkdir(logdir)
    checkpointer = keras.callbacks.ModelCheckpoint(
        os.path.join(logdir, 'model_epoch{epoch:02d}_valacc1{val_layers_softmax1_acc:.2f}_valacc2{val_layers_softmax2_acc:.2f}_valacc3{val_fc3_acc:.2f}.hdf5'),
        # os.path.join(logdir, 'model_epoch.hdf5'),
        verbose=1, save_weights_only=False)
    model.fit([atrain_time,atrain_space], [ay_train,ay_train,ay_train], epochs=epochs, validation_data=([avalid_time,avalid_space], [ay_valid,ay_valid,ay_valid]), batch_size=batch_size,
                        verbose=2, callbacks=[checkpointer])

    # , callbacks=[lrate], callbacks=[reduce_lr]
    return logdir



def test_model_and_find_top_acc(ax_test_time,ax_test_space, ay_test,all_modelpath,save_num,pad_mode,fram_num,fram_len,modell="model0"):
    path = all_modelpath
    save_num = save_num
    filename_list = os.listdir(path)

    # 遍历所有保存的模型
    res = []
    moidel_fin_save_dir = modell + pad_mode + "_" + str(fram_num) + "_" + str(fram_len)  # 当前文件夹下一个文件夹
    # print("save：", moidel_fin_save_dir)
    if not os.path.exists(moidel_fin_save_dir):  # 若该文件夹不存在 则创建一个
        os.mkdir(moidel_fin_save_dir)

    for i in range(len(filename_list)):
        saved_model = path + "/" + filename_list[i]
        print("test：", saved_model)
        model = keras.models.load_model(saved_model)
        test_acc = model.evaluate([ax_test_time,ax_test_space], [ay_test,ay_test,ay_test])
        test_acc_float = ('%.4f' % test_acc[-1])
        # 释放模型
        keras.backend.clear_session()
        # tf.compat.v1.reset_default_graph()
        tf.reset_default_graph()
        if len(res) <= save_num:
            res.append(test_acc_float)#加在末尾
            os.remove(saved_model)

        elif len(res) > save_num:

            res.sort(key=None, reverse=True)#排序降序
            nomber_num_acc = res[save_num - 1]
            if test_acc_float > nomber_num_acc:
                # 获取模型参数，便于建立新模型名
                name_no_hdf5 = saved_model[0:-5]  # 去除.hdf5
                name_no_hdf5 = name_no_hdf5.split("/")
                name_no_hdf5 = name_no_hdf5[1]
                str_list_name = name_no_hdf5.split("_")  # 分隔方式_

                # print(str_list_name)
                epoch_param = str_list_name[1]
                val_acc_param = str_list_name[4]
                new_modelname = modell + pad_mode + "_" + str(fram_num) + "_" + str(
                    fram_len) + "_" + epoch_param + "_" + val_acc_param + "_test_" + str(test_acc_float) + ".hdf5"
                # print(new_modelname)
                # print("save: ", new_modelname)
                savenew_modelname_path = moidel_fin_save_dir + "/" + new_modelname  # 保存路径
                # print(saved_model)
                print(savenew_modelname_path)
                shutil.move(saved_model, savenew_modelname_path)  # 移动并改名
            else:
                os.remove(saved_model)

            res.append(test_acc_float)

    newmodel_list = os.listdir(moidel_fin_save_dir)
    print(newmodel_list)
    finres = []
    for _ in newmodel_list:
        newname_no_hdf5 = _[0:-5]  # 去除.hdf5
        newstr_list_name = newname_no_hdf5.split("_")  # 分隔方式_
        print(newstr_list_name[-1])
        # print(moidel_fin_save_dir+"/"+_)
        if newstr_list_name[-1] < nomber_num_acc:
            os.remove(moidel_fin_save_dir + "/" + _)
        else:
            finres.append(newstr_list_name[-1])

    lastmodel_list = os.listdir(moidel_fin_save_dir)
    # print(lastmodel_list)

    return lastmodel_list


def aaaa(dataos,datanum,batch_size,data_list,lrate):
    adir0 = "a" + str(datanum) + "dir0"
    amodel0 = "a" + str(datanum) + "model0"
    bdir1 = "b" + str(datanum) + "dir1"
    bmodel1 = "b" + str(datanum) + "model1"

    for i in range(len(data_list)):
        data_file = dataos + "/" + data_list[i]
        pad_mode, fram_num, fram_len = get_mode_num_len(data_file)

        data_t, lable_all = get_data(data_file)
        data_0, data_1 = split_data(data_t, fram_num, fram_len)

        train_time,train_space, y_train, valid_time,valid_space, y_valid, test_time,test_space, y_test = train_test_split_and_scaler(data_0,
                                                                                                           lable_all,fram_num,fram_len)

        for llr in lrate:
            strlr = str(llr)
            strllr = strlr.split(".")[1]

            # bRDTCN-CNN
            modela = bulid_model_RDTCN_CNN(llr, fram_num, fram_len)
            logdira = adir0 + strllr + "bilstm"
            modella = amodel0 + strllr + "bilstm"
            logdir = train_save(modela, train_time,train_space, y_train,valid_time,valid_space, y_valid, epoch, fram_num, fram_len,
                                logdira, batch_size)
            print("l", logdir)
            lastmodel_list = test_model_and_find_top_acc(test_time, test_space,y_test, logdir, save_e, pad_mode, fram_num, fram_len,
                                                         modella)
            print(lastmodel_list)
            keras.backend.clear_session()
            tf.reset_default_graph()

        train_time1,train_space1, y_train1, valid_time1,valid_space1, y_valid1, test_time1,test_space1, y_test1 = train_test_split_and_scaler(data_1,
                                                                                                                 lable_all, fram_num, fram_len)

        for llr in lrate:
            strlr = str(llr)
            strllr = strlr.split(".")[1]

            # RDTCN-CNN
            modelb = bulid_model_RDTCN_CNN(llr, fram_num, fram_len)
            logdirb = bdir1 + strllr + "bilstm"
            modellb = bmodel1 + strllr + "bilstm"
            logdir1 = train_save(modelb, train_time1,train_space1, y_train1, valid_time1,valid_space1, y_valid1, epoch, fram_num, fram_len,
                                 logdirb, batch_size)
            print("l", logdir1)
            lastmodel_list = test_model_and_find_top_acc(test_time1,test_space1, y_test1, logdir1, save_e, pad_mode, fram_num,
                                                         fram_len, modellb)
            print(lastmodel_list)
            keras.backend.clear_session()
            tf.reset_default_graph()
    return 0


dataos500="数据"
dataos753="数据"

data_500_list = os.listdir(dataos500)
data_753_list = os.listdir(dataos753)
print(data_500_list)
print(data_753_list)

# 数据集500  六个模型  三个学习率  f0  f1   一共3*6*3*2
lrate=[0.001]
batch_size = 64
datanum = 500
epoch = 300

# aaaa(dataos500,500,batch_size, data_500_list, lrate)

# aaaa(dataos753,753,batch_size, data_753_list, lrate)




# ### 500+753
r = random.random
random.seed(1234)

adir0 = "a" + str(1253) + "dir0"
amodel0 = "a" + str(1253) + "model0"
bdir1 = "b" + str(1253) + "dir1"
bmodel1 = "b" + str(1253) + "model1"

for i in range(len(data_500_list)):
    data_file = dataos500+"/"+data_500_list[i]
    pad_mode, fram_num, fram_len = get_mode_num_len(data_file)

    data_file753 = "数据l753/lstm753next_fram_len_" + str(fram_len) + "_" + str(fram_num) + "_" + str(fram_len) + ".txt"
    data_t500, lable_all500 = get_data(data_file)
    data_t753, lable_all753 = get_data(data_file753)

    print(data_t753.shape)

    data_t = np.concatenate((data_t500, data_t753), axis=0)
    lable_all = np.concatenate((lable_all500, lable_all753), axis=0)
    data500753all = np.concatenate((data_t, lable_all), axis=1)

    random.shuffle(data500753all, random=r)
    data_t = data500753all[:, :-2]
    label_all1= data500753all[:, -2:]

    data_0, data_1 = split_data(data_t, fram_num, fram_len)

    train_time_all, train_space_all, y_train_all, valid_time_all, valid_space_all, y_valid_all, test_time_all, test_space_all, y_test_all = train_test_split_and_scaler(data_0,
        label_all1, fram_num, fram_len)

    for llr in lrate:
        strlr = str(llr)
        strllr = strlr.split(".")[1]

        # bilstm
        modela = bulid_model_RDTCN_CNN(llr, fram_num, fram_len)
        logdira = adir0 + strllr + "bilstm"
        modella = amodel0 + strllr + "bilstm"
        logdir = train_save(modela, train_time_all, train_space_all, y_train_all, valid_time_all, valid_space_all, y_valid_all, epoch, fram_num,fram_len,
                            logdira, batch_size)
        print("l", logdir)
        lastmodel_list = test_model_and_find_top_acc(test_time_all, test_space_all, y_test_all, logdir, save_e, pad_mode, fram_num,fram_len,
                                                     modella)
        print(lastmodel_list)
        keras.backend.clear_session()
        tf.reset_default_graph()

    train_time1_all, train_space1_all, y_train1_all, valid_time1_all, valid_space1_all, y_valid1_all, test_time1_all, test_space1_all, y_test1_all = train_test_split_and_scaler(data_1,
        label_all1, fram_num, fram_len)

    for llr in lrate:
        strlr = str(llr)
        strllr = strlr.split(".")[1]

        # bilstm
        modelb = bulid_model_RDTCN_CNN(llr, fram_num, fram_len)
        logdirb = bdir1 + strllr + "bilstm"
        modellb = bmodel1 + strllr + "bilstm"
        logdir1 = train_save(modelb, train_time1_all, train_space1_all, y_train1_all, valid_time1_all, valid_space1_all, y_valid1_all, epoch,
                             fram_num, fram_len,
                             logdirb, batch_size)
        print("l", logdir1)
        lastmodel_list = test_model_and_find_top_acc(test_time1_all, test_space1_all, y_test1_all, logdir1, save_e, pad_mode,
                                                     fram_num,
                                                     fram_len, modellb)
        print(lastmodel_list)
        keras.backend.clear_session()
        tf.reset_default_graph()


