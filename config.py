import argparse
def get_args() -> dict:
    parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')
    parser.add_argument('--device', type=str,  default='cuda:0')
    parser.add_argument('--seed', type=int, default=44)  # 'cuda:0'
    parser.add_argument('--inputs_path', type=str, default='E:/data/tianxu/')
    parser.add_argument('--nc_data_path', type=str, default='E:/')
    parser.add_argument('--product', type=str, default='SM_GD_NEW')
    parser.add_argument('--workname', type=str, default='SM_GD_NEW')
    parser.add_argument('--modelname', type=str,default='informer')
    parser.add_argument('--label', nargs='+', type=str, default=["volumetric_soil_water_layer_1"])
    parser.add_argument('--data_type', type=str, default='float32')
    parser.add_argument('--selected_year', nargs='+', type=int, default=[2015, 2020])  # 2015-2020
    parser.add_argument('--forcing_list', nargs='+', type=str,
                        default=["2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind",
                                 "precipitation", "surface_pressure", "specific_humidity"])
    parser.add_argument('--land_surface_list', nargs='+', type=str,
                        default=["surface_solar_radiation_downwards_w_m2", "surface_thermal_radiation_downwards_w_m2",
                                 "soil_temperature_level_1"])
    parser.add_argument('--static_list', nargs='+', type=str, default=["dem", "clay", "sand", "silt", "landtype"])
    parser.add_argument('--memmap', type=bool, default=True)
    parser.add_argument('--test_year', nargs='+', type=int, default=[2020])
    parser.add_argument('--spatial_resolution', type=float, default=0.1)
    parser.add_argument('--normalize', type=bool, default=True)
    parser.add_argument('--split_ratio', type=float, default=0.8)
    parser.add_argument('--valid_split', type=bool, default=True)
    parser.add_argument('--batch_size', type=float, default=512)
    parser.add_argument('--stride', type=float, default=1)
    # train arguments
    parser.add_argument('--epochs', type=float, default=10000,help='train epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
    parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    parser.add_argument('--num_repeat', type=float, default=1)
    # model
    parser.add_argument('--forcast_time',type=int,default='0' ,help='the prediction time of lable ')
    parser.add_argument('--input_size', type=int, default=14)
    parser.add_argument('--out_size', type=int, default=14)  # 预报天数
    parser.add_argument('--num_layers', type=int, default=2)  #
    parser.add_argument('--normalize_type', type=str, default='region')  # global, #region
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--seq_len', type=int, default=7, help='input sequence length of Informer encoder')
    parser.add_argument('--label_len', type=int, default=48, help='start token length of Informer decoder')
    parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
    # Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
    parser.add_argument('--padding', type=int, default=0, help='padding type')
    parser.add_argument('--distil', action='store_false',help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    cfg = vars(parser.parse_args())
    return cfg
