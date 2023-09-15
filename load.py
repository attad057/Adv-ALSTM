from datetime import datetime
from lib2to3.pgen2.pgen import DFAState
import numpy as np
import os
import pandas as pd

def load_cla_data(data_path, tra_date, val_date, tes_date, seq=2,
                  date_format='%Y-%m-%d', fea_dim  = None, return_mappings = False):
    fnames = [fname for fname in os.listdir(data_path) if
              os.path.isfile(os.path.join(data_path, fname))]
    print(len(fnames), ' tickers selected')

    data_EOD = []
    for index, fname in enumerate(fnames):
        # print(fname)
        single_EOD = np.genfromtxt(
            os.path.join(data_path, fname), dtype=float, delimiter=',',
            skip_header=False
        )
        # print('data shape:', single_EOD.shape)
        data_EOD.append(single_EOD)

    default_fea_dim = data_EOD[0].shape[1] - 2
    if fea_dim == None:
        fea_dim = default_fea_dim

    trading_dates = np.genfromtxt(
        os.path.join(data_path, '..', 'trading_dates.csv'), dtype=str,
        delimiter=',', skip_header=False
    )
    print(len(trading_dates), 'trading dates:')

    # transform the trading dates into a dictionary with index, at the same
    # time, transform the indices into a dictionary with weekdays
    dates_index = {}
    # indices_weekday = {}
    data_wd = np.zeros([len(trading_dates), 5], dtype=float)
    wd_encodings = np.identity(5, dtype=float)
    for index, date in enumerate(trading_dates):
        dates_index[date] = index
        # indices_weekday[index] = datetime.strptime(date, date_format).weekday()
        data_wd[index] = wd_encodings[datetime.strptime(date, date_format).weekday()]

    tra_ind = dates_index[tra_date]
    val_ind = dates_index[val_date]
    tes_ind = dates_index[tes_date]
    print(tra_ind, val_ind, tes_ind)

    # count training, validation, and testing instances
    tra_num = 0
    val_num = 0
    tes_num = 0
    # training
    for date_ind in range(tra_ind, val_ind):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if abs(data_EOD[tic_ind][date_ind][-2]) > 1e-8:
                if data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                    tra_num += 1
    print(tra_num, ' training instances')

    # validation
    for date_ind in range(val_ind, tes_ind):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if abs(data_EOD[tic_ind][date_ind][-2]) > 1e-8:
                if data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                    val_num += 1
    print(val_num, ' validation instances')

    # testing
    for date_ind in range(tes_ind, len(trading_dates)):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if abs(data_EOD[tic_ind][date_ind][-2]) > 1e-8:
                if data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                    tes_num += 1
    print(tes_num, ' testing instances')

    # generate training, validation, and testing instances
    # training
    tra_pv = np.zeros([tra_num, seq, fea_dim], dtype=float)
    tra_wd = np.zeros([tra_num, seq, 5], dtype=float)
    tra_gt = np.zeros([tra_num, 1], dtype=float)
    tes_mappings = []
    val_mappings = []

    ins_ind = 0
    for date_ind in range(tra_ind, val_ind):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if abs(data_EOD[tic_ind][date_ind][-2]) > 1e-8 and \
                    data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                tra_pv[ins_ind] = data_EOD[tic_ind][date_ind - seq: date_ind, : -(default_fea_dim - fea_dim + 2)]
                tra_wd[ins_ind] = data_wd[date_ind - seq: date_ind, :]
                tra_gt[ins_ind, 0] = (data_EOD[tic_ind][date_ind][-2] + 1) / 2
                ins_ind += 1

    # validation
    val_pv = np.zeros([val_num, seq, fea_dim], dtype=float)
    val_wd = np.zeros([val_num, seq, 5], dtype=float)
    val_gt = np.zeros([val_num, 1], dtype=float)
    val_ins_ind = []
    ins_ind = 0
    for date_ind in range(val_ind, tes_ind):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):
            if len(val_ins_ind) < tic_ind + 1:
                val_ins_ind.append([])
            if abs(data_EOD[tic_ind][date_ind][-2]) > 1e-8 and \
                            data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                val_pv[ins_ind] = data_EOD[tic_ind][date_ind - seq: date_ind, :-(default_fea_dim - fea_dim + 2)]
                val_wd[ins_ind] = data_wd[date_ind - seq: date_ind, :]
                val_gt[ins_ind, 0] = (data_EOD[tic_ind][date_ind][-2] + 1) / 2
                if return_mappings == True:
                    #prev_date = max((x['date'] for x in list(filter(lambda x: x['ticker_filename'] == fnames[tic_ind], val_mappings))), default=None)

                    length = len(val_ins_ind[tic_ind])
                    val_n_adj_close = data_EOD[tic_ind][date_ind][4]
                    val_adj_close = data_EOD[tic_ind][date_ind][-1]
                    prev_val_adj_close =  data_EOD[tic_ind][date_ind-1][-1]
                    val_log_return = np.log(val_adj_close / prev_val_adj_close)

                    val_mappings.append({
                        'ins_ind': ins_ind,
                        'ticker_filename': fnames[tic_ind],
                        'date': trading_dates[date_ind],
                        'n_adj_close': val_n_adj_close,
                        'gt': val_gt[ins_ind, 0],
                        'adj_close': val_adj_close,
                        'prev_adj_close': prev_val_adj_close,
                        'log_return': val_log_return,
                        'prev_ins_ind': None if length == 0 else val_ins_ind[tic_ind][length - 1]
                    })
                    val_ins_ind[tic_ind].append(ins_ind)
                ins_ind += 1

    # testing
    tes_pv = np.zeros([tes_num, seq, fea_dim], dtype=float)
    tes_wd = np.zeros([tes_num, seq, 5], dtype=float)
    tes_gt = np.zeros([tes_num, 1], dtype=float)
    tes_ins_ind = []
    ins_ind = 0
    for date_ind in range(tes_ind, len(trading_dates)):
        # filter out instances without length enough history
        if date_ind < seq:
            continue
        for tic_ind in range(len(fnames)):                          
            if len(tes_ins_ind) < tic_ind + 1:
                tes_ins_ind.append([])
            if abs(data_EOD[tic_ind][date_ind][-2]) > 1e-8 and \
                            data_EOD[tic_ind][date_ind - seq: date_ind, :].min() > -123320:
                tes_pv[ins_ind] = data_EOD[tic_ind][date_ind - seq: date_ind, :-(default_fea_dim - fea_dim + 2)]
                # # for the momentum indicator
                # tes_pv[ins_ind, -1, -1] = data_EOD[tic_ind][date_ind - 1, -1] - data_EOD[tic_ind][date_ind - 11, -1]
                tes_wd[ins_ind] = data_wd[date_ind - seq: date_ind, :]
                tes_gt[ins_ind, 0] = (data_EOD[tic_ind][date_ind][-2] + 1) / 2
                if return_mappings == True:
                    #prev_date = max((x['date'] for x in list(filter(lambda x: x['ticker_filename'] == fnames[tic_ind], tes_mappings))), default=None)

                    length = len(tes_ins_ind[tic_ind])
                    tes_n_adj_close = data_EOD[tic_ind][date_ind][4]
                    tes_adj_close = data_EOD[tic_ind][date_ind][-1]
                    prev_tes_adj_close =  data_EOD[tic_ind][date_ind-1][-1]
                    tes_log_return = np.log(tes_adj_close / prev_tes_adj_close)
                    # if (tes_n_adj_close > 0 and tes_log_return < 0) or (tes_n_adj_close < 0 and tes_log_return > 0):
                    #     tes_log_return = -tes_log_return

                    tes_mappings.append({
                        'ins_ind': ins_ind,
                        'ticker_filename': fnames[tic_ind],
                        'date': trading_dates[date_ind],
                        'n_adj_close': tes_n_adj_close,
                        'gt': tes_gt[ins_ind, 0],
                        'adj_close': tes_adj_close,
                        'prev_adj_close': prev_tes_adj_close,
                        'log_return': tes_log_return,
                        'prev_ins_ind': None if length == 0 else tes_ins_ind[tic_ind][length - 1]
                    })
                    tes_ins_ind[tic_ind].append(ins_ind)
                ins_ind += 1

    if return_mappings == False:
        return tra_pv, tra_wd, tra_gt, val_pv, val_wd, val_gt, tes_pv, tes_wd, tes_gt
    else:
        return tra_pv, tra_wd, tra_gt, val_pv, val_wd, val_gt, tes_pv, tes_wd, tes_gt, val_mappings, tes_mappings

if __name__ == '__main__':
    # TEST
    _, _, tra_gt, _, _, val_gt, _, _, tes_gt = load_cla_data(
        '/home/ffl/nus/MM/fintech/tweet_stock/data/stocknet-dataset/price/ourpped',
        '2014-01-02', '2015-08-03', '2015-10-01'
    )
    print(np.sum(tra_gt))
    print(np.sum(val_gt))
    print(np.sum(tes_gt))
    print(np.sum(tes_gt) / 3720)