import time
import numpy as np
import torch
import torch.nn
from tqdm import trange
from data_gen import load_test_data_for_rnn,load_train_data_for_rnn,load_test_data_for_cnn, load_train_data_for_cnn,erath_data_transform,sea_mask_rnn,sea_mask_cnn,make_train_data
from loss import  NaNMSELoss
from models.model_informer import Informer


def train(x,
          y,
          static,
          mask, 
          scaler_x,
          scaler_y,
          cfg,
          num_repeat,
          PATH,
          out_path,
          device,
          num_task=None):
   
    patience = cfg['patience']
    wait = 0
    best = 9999
    valid_split=cfg['valid_split']
    print('the device is {d}'.format(d=device))
    print('y type is {d_p}'.format(d_p=y.dtype))
    print('static type is {d_p}'.format(d_p=static.dtype))
    if valid_split:
        nt,nf,nlat,nlon = x.shape  #x shape :nt,nf,nlat,nlon
        N = int(nt*cfg['split_ratio'])
        x_valid, y_valid, static_valid = x[N:], y[N:], static
        x, y = x[:N], y[:N]

    lossmse = torch.nn.MSELoss().to(device)
#	filter Antatctica
    print('x_train shape is', x.shape)
    print('y_train shape is', y.shape)
    print('static_train shape is', static.shape)
    print('mask shape is', mask.shape)
    print(np.isnan(x_valid).any(), np.isnan(y_valid).any())
    print(np.isnan(x).any(), np.isnan(y).any())
    # mask sea regions
    #Determine the land boundary
    if cfg['modelname'] in ['LSTM','EDLSTM','AttLSTM','informer']:
        if valid_split:
            x_valid = np.where(np.isnan(x_valid) & (mask[np.newaxis, :, :, np.newaxis] == 1), np.nanmean(x_valid), x_valid)
            static_valid = np.where(np.isnan(static_valid) & (mask[:, :, np.newaxis] == 1), np.nanmean(static_valid), static_valid)
            x_valid, y_valid, static_valid = sea_mask_rnn(cfg, x_valid, y_valid, static_valid, mask)
        x = np.where(np.isnan(x) & (mask[np.newaxis, :, :, np.newaxis] == 1), np.nanmean(x), x)
        static = np.where(np.isnan(static) & (mask[ :, :, np.newaxis] == 1), np.nanmean(static), static)
        y = np.where(np.isnan(y) & (mask[np.newaxis, :, :, np.newaxis] == 1),np.nanmean(y) , y)
        x, y, static = sea_mask_rnn(cfg, x, y, static, mask)
        print(np.isnan(x_valid).any(), np.isnan(y_valid).any())
        print(np.isnan(x).any(), np.isnan(y).any())
        x,y, static = make_train_data(cfg, x, y, static)
    print(x.shape, y.shape, static.shape)
    for num_ in range(cfg['num_repeat']):
        if cfg['modelname'] in ['informer']:
            model = Informer(cfg['enc_in'], cfg['dec_in'], cfg['c_out'], cfg['seq_len'], cfg['label_len'], cfg['out_size']).to(device)

	 # Prepare for training
    # NOTE: Only use `Adam`, we didn't apply adaptively
    # learing rate schedule. We found `Adam` perform
    # much better than `Adagrad`, `Adadelta`.
        optim = torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'])

        with trange(1, cfg['epochs']+1) as pbar:
            for epoch in pbar:
                pbar.set_description(
                    cfg['modelname']+' '+str(num_repeat))
                t_begin = time.time()
                # train
                MSELoss = 0
                shuffle_index = torch.randperm(x.shape[1])
                x = x[:, shuffle_index, :, :]
                y = y[:, shuffle_index, :]
 # ------------------------------------------------------------------------------------------------------------------------------
 #  train way for LSTM model
                if cfg["modelname"] in \
                            ['informer']:
                        # generate batch data for Recurrent Neural Network
                        for i in range(0,x.shape[1]):
                            x_batch, y_batch, aux_batch = \
                            load_train_data_for_rnn(cfg, x[:,i,:,:], y[:,i,:], static[:,i,:], scaler_y)
                            x_batch = torch.from_numpy(x_batch).to(device)
                            aux_batch = aux_batch.reshape(aux_batch.shape[0],1,aux_batch.shape[1])
                            aux_batch = torch.from_numpy(aux_batch).to(device)
                            y_batch = torch.from_numpy(y_batch).to(device)
                            aux_batch = aux_batch.repeat(1,x_batch.shape[1],1)
                            x_batch = torch.cat([x_batch, aux_batch], 2).to(device)
                            pred = model(x_batch)
                            pred = torch.squeeze(pred,1).to(device)
                            loss = NaNMSELoss.fit(cfg, pred.float(), y_batch.float(), lossmse)
                            optim.zero_grad()
                            loss.backward()
                            optim.step()
                            MSELoss += loss.item()
 # ------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------
                t_end = time.time()
                # get loss log
                loss_str = "Epoch {} Train MSE Loss {:.3f} time {:.2f}".format(epoch, MSELoss , t_end - t_begin)
                print(loss_str)
                # validate
		#Use validation sets to test trained models
		#If the error is smaller than the minimum error, then save the model.
                if valid_split:
                    del x_batch, y_batch, aux_batch
                    MSE_valid_loss = 0
                    if epoch % 20 == 0:
                        wait += 1
                        # NOTE: We used grids-mean NSE as valid metrics.
                        t_begin = time.time()
# ------------------------------------------------------------------------------------------------------------------------------
 #  validate way for LSTM model
                        if cfg["modelname"] in ['informer']:
                            gt_list = [i for i in range(0,x_valid.shape[0]-cfg['seq_len'],cfg["stride"])]
                            x_valid_new, y_valid_new, aux_valid_new = \
                         load_test_data_for_rnn(cfg, x_valid, y_valid, static_valid)
                            for i in range(0, x_valid_new.shape[1]):
                                x_valid_batch = torch.Tensor(x_valid_new[:,i,:,:]).to(device)
                                y_valid_batch = torch.Tensor(y_valid_new[:,i,:]).to(device)
                                aux_valid_batch = aux_valid_new[:,i,:]
                                aux_valid_batch = aux_valid_batch.reshape(aux_valid_batch.shape[0],1,aux_valid_batch.shape[1])
                                aux_valid_batch = torch.Tensor(aux_valid_batch).to(device)
                                aux_valid_batch = aux_valid_batch.repeat(1,x_valid_batch.shape[1],1)
                                x_valid_batch = torch.cat([x_valid_batch, aux_valid_batch], 2).to(device)
                                with torch.no_grad():
                                    pred_valid = model(x_valid_batch)
                                mse_valid_loss = NaNMSELoss.fit(cfg, pred_valid.squeeze(1), y_valid_batch,lossmse)
                                MSE_valid_loss += mse_valid_loss.item()
# ------------------------------------------------------------------------------------------------------------------------------
             

                        t_end = time.time()
                        mse_valid_loss = MSE_valid_loss/(len(gt_list))
                        # get loss log
                        loss_str = '\033[1;31m%s\033[0m' % \
                                "Epoch {} Val MSE Loss {:.3f}  time {:.2f}".format(epoch,mse_valid_loss, 
                                    t_end-t_begin)
                        print(loss_str)
                        val_save_acc = mse_valid_loss

                        # save best model by val loss
                        # NOTE: save best MSE results get `single_task` better than `multi_tasks`
                        #       save best NSE results get `multi_tasks` better than `single_task`
                        if val_save_acc < best:
                        #if MSE_valid_loss < best:
                            torch.save(model,out_path+cfg['modelname']+'_para.pkl')
                            wait = 0  # release wait
                            best = val_save_acc #MSE_valid_loss
                            print('\033[1;31m%s\033[0m' % f'Save Epoch {epoch} Model')
                else:
                    # save best model by train loss
                    if MSELoss < best:
                        best = MSELoss
                        wait = 0
                        torch.save(model,out_path+cfg['modelname']+'_para.pkl')
                        print('\033[1;31m%s\033[0m' % f'Save Epoch {epoch} Model')

                # early stopping
                if wait >= patience:
                    return
            return

