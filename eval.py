import numpy as np
import torch
import time
from data_gen import erath_data_transform
import sys
from data import Dataset

def batcher_lstm(x_test, y_test, aux_test, seq_len,forcast_time):
    n_t, n_feat = x_test.shape
    n = (n_t-seq_len-forcast_time-14)
    x_new = np.zeros((n, seq_len, n_feat))*np.nan
    y_new = np.zeros((n,14))*np.nan
    aux_new = np.zeros((n,aux_test.shape[0]))*np.nan


    for i in range(n):
        x_new[i] = x_test[i:i+seq_len]
        y_new[i] = y_test[i+seq_len+forcast_time:i+seq_len+forcast_time+14,-1]
        aux_new[i] = aux_test
    return x_new, y_new, aux_new

def batcher_cnn(x_test, y_test, aux_test, seq_len,forcast_time,spatial_offset,i,j,lat_index,lon_index):
    x_test = x_test.transpose(0,3,1,2)
    y_test = y_test.transpose(0,3,1,2)
    aux_test = aux_test.transpose(2,0,1)
    n_t, n_feat, n_lat,n_lon = x_test.shape

    n = (n_t-seq_len-forcast_time-14)
    x_new = np.zeros((n, seq_len, n_feat,2*spatial_offset+1,2*spatial_offset+1))*np.nan
    y_new = np.zeros((n,1))*np.nan
    aux_new = np.zeros((n,aux_test.shape[0],2*spatial_offset+1,2*spatial_offset+1))*np.nan
    for ni in range(n):
        lat_index_bias = lat_index[i] + spatial_offset
        lon_index_bias = lon_index[j] + spatial_offset
        x_new[ni] = x_test[ni:ni+seq_len,:,lat_index[lat_index_bias-spatial_offset:lat_index_bias+spatial_offset+1],:][:,:,:,lon_index[lon_index_bias-spatial_offset:lon_index_bias+spatial_offset+1]]
        y_new[ni] = y_test[ni+seq_len+forcast_time:ni+seq_len+forcast_time+14,:,i,j]
        aux_new[ni] = aux_test[:,lat_index[lat_index_bias-spatial_offset:lat_index_bias+spatial_offset+1],:][:,:,lon_index[lon_index_bias-spatial_offset:lon_index_bias+spatial_offset+1]]
    return x_new, y_new, aux_new

def batcher_convlstm(x_test, y_test, aux_test, seq_len,forcast_time,spatial_offset,i,j,lat_index,lon_index):
    x_test = x_test.transpose(0,3,1,2)
    y_test = y_test.transpose(0,3,1,2)
    aux_test = aux_test.transpose(2,0,1)
    n_t, n_feat, n_lat,n_lon = x_test.shape

    n = (n_t-seq_len-forcast_time-14)
    x_new = np.zeros((n, seq_len, n_feat,2*spatial_offset+1,2*spatial_offset+1))*np.nan
    y_new = np.zeros((n,14,1))*np.nan
    aux_new = np.zeros((n,aux_test.shape[0],2*spatial_offset+1,2*spatial_offset+1))*np.nan
    for ni in range(n):
        lat_index_bias = lat_index[i] + spatial_offset
        lon_index_bias = lon_index[j] + spatial_offset
        x_new[ni] = x_test[ni:ni+seq_len,:,lat_index[lat_index_bias-spatial_offset:lat_index_bias+spatial_offset+1],:][:,:,:,lon_index[lon_index_bias-spatial_offset:lon_index_bias+spatial_offset+1]]
        y_new[ni] = y_test[ni+seq_len+forcast_time:ni+seq_len+forcast_time+14,:,i,j]
        aux_new[ni] = aux_test[:,lat_index[lat_index_bias-spatial_offset:lat_index_bias+spatial_offset+1],:][:,:,lon_index[lon_index_bias-spatial_offset:lon_index_bias+spatial_offset+1]]
    return x_new, y_new, aux_new


def test(x, y, static, scaler, cfg, model,device):
    cls = Dataset(cfg)          
    model.eval()
    if cfg['modelname'] in ['CNN', 'ConvLSTM','AttConvLSTM','EDConvLSTM','AEDConvLSTM']:
#	Splice x according to the sphere shape
        lat_index,lon_index = erath_data_transform(cfg, x)
        print('\033[1;31m%s\033[0m' % "Applied Model is {m_n}, we need to transform the data according to the sphere shape".format(m_n=cfg['modelname']))


    y_pred_ens = np.zeros((y.shape[0]-cfg["seq_len"]-cfg['forcast_time']-14,14, y.shape[1], y.shape[2]))*np.nan
    y_true = np.zeros((y.shape[0]-cfg["seq_len"]-cfg['forcast_time']-14,14, y.shape[1], y.shape[2]))*np.nan
    for i in range(y_true.shape[0]):
        y_true[i] = y[cfg["seq_len"]+cfg['forcast_time']+i:cfg["seq_len"]+cfg['forcast_time']+i+14,:,:,-1]

# ------------------------------------------------------------------------------------------------------------------------------              
    # for each grid by lstm model
    if cfg["modelname"] in ['LSTM','EDLSTM','AttLSTM']:
        count = 1
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                x_new, y_new, static_new = batcher_lstm(x[:,i,j,:], y[:,i,j,:], static[i,j,:], cfg["seq_len"],cfg['forcast_time']) # 对于每个格点制作时间序列
                x_new = torch.from_numpy(x_new).to(device)
                static_new = torch.from_numpy(static_new).to(device)
                static_new = static_new.unsqueeze(1)
                static_new = static_new.repeat(1,x_new.shape[1],1)
                x_new = torch.cat([x_new, static_new], 2).to(device)
                x_new = x_new.to(torch.float32).to(device)
                pred = model(x_new)
                pred = pred.cpu().detach().numpy()
                pred = np.squeeze(pred)

                if cfg["normalize"] and cfg['normalize_type'] in ['region']:
                    pred = cls.reverse_normalize(pred,'output',scaler[:,i,j,0],'minmax',-1)
                elif cfg["normalize"] and cfg['normalize_type'] in ['global']:
                    pred = cls.reverse_normalize(pred,'output',scaler,'minmax',-1)

                y_pred_ens[:,:,i,j]=pred
                if count % 1000 == 0:
                    print(time.strftime("%Y-%m-%d %H:%M:%S",time.gmtime()))
                    print('\r',end="")                    
                    print('Remain {fs} thousand(s) predictions'.format(fs=(x.shape[1]*x.shape[2]-count)/1000))
                    sys.stdout.flush()
                time.sleep(0.0001)
                count = count+1
    # for each grid by cnn model
    if cfg["modelname"] in ['CNN']:
        count = 1
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                x_new, y_new, static_new = batcher_convlstm(x, y, static, cfg["seq_len"], cfg['forcast_time'],
                                                            cfg["spatial_offset"], i, j, lat_index, lon_index)
                x_new = np.nan_to_num(x_new)
                static_new = np.nan_to_num(static_new)
                x_new = torch.from_numpy(x_new).to(device)
                static_new = torch.from_numpy(static_new).to(device)
                # x_new = torch.cat([x_new, static_new], 1)
                x_new = x_new.squeeze(1)
                x_new = x_new.reshape(x_new.shape[0], x_new.shape[1] * x_new.shape[2], x_new.shape[3],
                                      x_new.shape[4])
                x_new = torch.cat([x_new, static_new], 1).to(device)
                pred = model(x_new)

                pred = pred.cpu().detach().numpy()
                pred = np.squeeze(pred)
                if cfg["normalize"] and cfg['normalize_type'] in ['region']:
                    pred = cls.reverse_normalize(pred, 'output', scaler[:, i, j, 0], 'minmax', -1)
                elif cfg["normalize"] and cfg['normalize_type'] in ['global']:
                    pred = cls.reverse_normalize(pred, 'output', scaler, 'minmax', -1)
                y_pred_ens[:, :,i, j] = pred
                if count % 1000 == 0:
                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
                    print('\r', end="")
                    print('Remain {fs} thound predictions'.format(fs=(x.shape[1] * x.shape[2] - count) / 1000))
                    sys.stdout.flush()
                time.sleep(0.0001)
                count = count + 1
    # ------------------------------------------------------------------------------------------------------------------------------
    # for each grid by convlstm model
    if cfg["modelname"] in ['ConvLSTM','AttConvLSTM','EDConvLSTM','AEDConvLSTM']:
        count = 1
        for i in range(x.shape[1]):
            for j in range(x.shape[2]):
                x_new, y_new, static_new = batcher_convlstm(x, y, static, cfg["seq_len"], cfg['forcast_time'],
                                                            cfg["spatial_offset"], i, j, lat_index, lon_index)
                x_new = np.nan_to_num(x_new)
                static_new = np.nan_to_num(static_new)
                x_new = torch.from_numpy(x_new).to(device)
                static_new = torch.from_numpy(static_new).to(device)
                static_new = static_new.unsqueeze(1)
                static_new = static_new.repeat(1, x_new.shape[1], 1, 1, 1)
                #print(x_new.shape)
                #print(static_new.shape)
                x_new = torch.cat([x_new, static_new], 2).to(device)
                pred = model(x_new.float())
                pred = pred.cpu().detach().numpy()
                pred = np.squeeze(pred)
                if cfg["normalize"] and cfg['normalize_type'] in ['region']:
                    pred = cls.reverse_normalize(pred, 'output', scaler[:, i, j, 0], 'minmax', -1)
                elif cfg["normalize"] and cfg['normalize_type'] in ['global']:
                    pred = cls.reverse_normalize(pred, 'output', scaler, 'minmax', -1)
                y_pred_ens[:,:, i, j] = pred
                if count % 1000 == 0:
                    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
                    print('\r', end="")
                    print('Remain {fs} thound predictions'.format(fs=(x.shape[1] * x.shape[2] - count) / 1000))
                    sys.stdout.flush()
                time.sleep(0.0001)
                count = count + 1

# ----------------------------------------------------------------------------------------------------------------------------              
    t_end = time.time()
    print('y_pred_ens shape is',y_pred_ens.shape)
    print('y_true shape is',y_true.shape)
    print('scaler shape is',scaler.shape)
 
    return y_pred_ens,y_true



