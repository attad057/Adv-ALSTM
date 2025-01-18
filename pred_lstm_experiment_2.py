import argparse
import copy
import numpy as np
import os
import random
from sklearn.utils import shuffle
import tensorflow as tf
from time import time
import pandas as pd
import dataframe_image as dfi
from scipy.stats import entropy
import pickle  
from pred_lstm import AWLSTM             
from load import load_cla_data
from evaluator import evaluate, label

try:
    from tensorflow.python.ops.nn_ops import leaky_relu
except ImportError:
    from tensorflow.python.framework import ops
    from tensorflow.python.ops import math_ops


    def leaky_relu(features, alpha=0.2, name=None):
        with ops.name_scope(name, "LeakyRelu", [features, alpha]):
            features = ops.convert_to_tensor(features, name="features")
            alpha = ops.convert_to_tensor(alpha, name="alpha")
            return math_ops.maximum(alpha * features, features)

def get_run_aggregate(val_mappings_df, tes_mappings_df,
                        best_valid_perf, best_test_perf,
                        cur_valid_perf, cur_tes_perf,
                        val_best_benchmark_model, tes_best_benchmark_model, 
                        method, dataset, dropout, run, p, prob_measure):
    df = None
    if val_mappings_df is not None:
        val_mappings_df['log_return_action'] = val_mappings_df['log_return'] * val_mappings_df['day_action']
        val_mappings_df['log_return_action_pred'] = val_mappings_df['log_return'] * val_mappings_df['day_action_pred']
        val_mappings_df['return_action'] = val_mappings_df['return']  * val_mappings_df['day_action']
        val_mappings_df['return_action_pred'] = val_mappings_df['return']  * val_mappings_df['day_action_pred']
        
        val_pre_returns =  val_mappings_df['return_action'].sum()
        val_pre_returns_pred =  val_mappings_df['return_action_pred'].sum()
        val_pre_log_returns = val_mappings_df['log_return_action'].sum()
        
        val_not_matches_gt_count = val_mappings_df[(val_mappings_df['matches_gt_day'] == 0)].shape[0]
        val_matches_gt_count = val_mappings_df[(val_mappings_df['matches_gt_day'] == 1)].shape[0]
        val_matches_gt_acc = val_matches_gt_count / (val_matches_gt_count + val_not_matches_gt_count)
        
        #val_pre_returns_avg = val_mappings_df['log_return_action'].mean()
        val_pre_returns_avg = val_mappings_df['return_action'].mean()

        #val_sharp_ratio = val_pre_returns_avg / val_mappings_df['log_return_action'].std()
        val_sharp_ratio = val_pre_returns_avg / val_mappings_df['return_action'].std()

        val_total_trading_days_skipped = val_mappings_df[(val_mappings_df['day_action'] == 0)].shape[0]
        val_total_trading_days_successful = val_mappings_df[(val_mappings_df['return_action'] > 0)].shape[0]
        total_val_trading_days = val_mappings_df.shape[0]

        val_total_trading_days_correctly_skipped = val_mappings_df[(val_mappings_df['day_action'] == 0) & (val_mappings_df['return_action_pred'] < 0)].shape[0]
        ratio_val_skipped_over_total = 1 if val_total_trading_days_skipped == 0 else val_total_trading_days_correctly_skipped / val_total_trading_days_skipped
        val_total_trading_days_count_incorrect_pred = val_mappings_df[(val_mappings_df['return_action_pred'] < 0)].shape[0]
        val_total_trading_days_incorrect_pred_not_skipped = val_mappings_df[(val_mappings_df['return_action'] < 0) & (val_mappings_df['return_action_pred'] < 0)].shape[0]
        val_total_trading_days_incorrect_pred_not_skipped_and_prob_0 = val_mappings_df[(val_mappings_df['return_action'] < 0) & (val_mappings_df['return_action_pred'] < 0) & (val_mappings_df['prob'] == 0)].shape[0]

        val_total_trading_days_correctly_skipped_exp1 = val_mappings_df[(val_mappings_df['day_action'] == 0) & (val_mappings_df['gt'] != val_mappings_df['day_action_pred'])]

        #val_profit_per_trade = val_pre_returns / (total_val_trading_days - val_total_trading_days_skipped)
        val_profit_per_trade = val_pre_returns / (total_val_trading_days - val_total_trading_days_skipped)

        perf_dict = {
                'method': [method],
                'dataset': [dataset],
                'dropout': [dropout],
                'run': [run],
                'prob': [p],
                'prob measure': [prob_measure],
                'total val trading days skipped': [val_total_trading_days_skipped],
                'total val trading days profitable': [val_total_trading_days_successful],
                'total val trading days averted loss by skipping': [val_total_trading_days_correctly_skipped],
                'val trading days averted loss by skipping percentage of total skipped': [ratio_val_skipped_over_total * 100],
                'total val trading days':[total_val_trading_days - val_total_trading_days_skipped],
                'total val return': [val_pre_returns],
                'total val return if without filtering': [val_pre_returns_pred],
                'total val log return': [val_pre_log_returns],
                'avg val return': [val_pre_returns_avg],
                'val profit per trade': [val_profit_per_trade],
                'val sharpe ratio': [val_sharp_ratio],
                'dropout val acc': [best_valid_perf['acc'] * 100],
                'dropout val mcc': [best_valid_perf['mcc']],
                'val acc': [cur_valid_perf['acc'] * 100],
                'val mcc': [cur_valid_perf['mcc']],
                'val acc of y to log return' : [val_matches_gt_acc * 100],
                'best benchmark val total return': [val_best_benchmark_model['total val return']], 
                'best benchmark val total profit per trade': [val_best_benchmark_model['total val profit per trade']],
                'best benchmark val avg sharpe ratio': [val_best_benchmark_model['val sharpe ratio']], 
                'best benchmark val total log return': [val_best_benchmark_model['total tes log return']],
                'best benchmark val acc': [val_best_benchmark_model['val accuracy']]                                                                                  
            }
        
        df = pd.DataFrame(perf_dict)

    df_2 = None
    if tes_mappings_df is not None:
        tes_mappings_df['log_return_action'] = tes_mappings_df['log_return'] * tes_mappings_df['day_action']
        tes_mappings_df['log_return_action_pred'] = tes_mappings_df['log_return'] * tes_mappings_df['day_action_pred']
        tes_mappings_df['return_action'] = tes_mappings_df['return']  * tes_mappings_df['day_action']
        tes_mappings_df['return_action_pred'] = tes_mappings_df['return']  * tes_mappings_df['day_action_pred']
              
        tes_pre_returns_pred = tes_mappings_df['return_action_pred'].sum()
        tes_pre_returns = tes_mappings_df['return_action'].sum()
        tes_pre_log_returns = tes_mappings_df['log_return_action'].sum()

        tes_not_matches_gt_day_count = tes_mappings_df[(tes_mappings_df['matches_gt_day'] == 0)].shape[0]
        tes_matches_gt_day_count = tes_mappings_df[(tes_mappings_df['matches_gt_day'] == 1)].shape[0]
        tes_matches_gt_acc = tes_matches_gt_day_count / (tes_matches_gt_day_count + tes_not_matches_gt_day_count)

        #tes_pre_returns_avg = tes_mappings_df['log_return_action'].mean()
        tes_pre_returns_avg = tes_mappings_df['return_action'].mean()

        #tes_sharp_ratio = tes_pre_returns_avg / tes_mappings_df['log_return_action'].std()
        tes_sharp_ratio = tes_pre_returns_avg / tes_mappings_df['return_action'].std()

        tes_total_trading_days_skipped = tes_mappings_df[(tes_mappings_df['day_action'] == 0)].shape[0]
        tes_total_trading_days_successful = tes_mappings_df[(tes_mappings_df['return_action'] > 0)].shape[0]
        total_tes_trading_days = tes_mappings_df.shape[0]
        tes_total_trading_days_correctly_skipped = tes_mappings_df[(tes_mappings_df['day_action'] == 0) & (tes_mappings_df['return_action_pred'] < 0)].shape[0]
        ratio_tes_skipped_over_total = 1 if tes_total_trading_days_skipped == 0 else tes_total_trading_days_correctly_skipped / tes_total_trading_days_skipped
        
        #tes_profit_per_trade = tes_pre_returns / (total_tes_trading_days - tes_total_trading_days_skipped)
        tes_profit_per_trade = tes_pre_returns / (total_tes_trading_days - tes_total_trading_days_skipped)

        perf_dict_2 = {
                'method': [method],
                'dataset': [dataset],
                'dropout': [dropout],
                'run': [run],
                'prob': [p],
                'prob measure': [prob_measure],
                'total tes trading days skipped': [tes_total_trading_days_skipped],
                'total tes trading days profitable': [tes_total_trading_days_successful],
                'total tes trading days averted loss by skipping': [tes_total_trading_days_correctly_skipped],
                'tes trading days averted loss by skipping percentage of total skipped': [ratio_tes_skipped_over_total * 100],
                'total tes trading days':[total_tes_trading_days - tes_total_trading_days_skipped],
                'total tes return': [tes_pre_returns],
                'total tes return if without filtering': [tes_pre_returns_pred],
                'total tes log return': [tes_pre_log_returns],
                'avg tes return': [tes_pre_returns_avg],
                'tes profit per trade': [tes_profit_per_trade],
                'tes sharpe ratio': [tes_sharp_ratio],
                'dropout tes acc': [best_test_perf['acc'] * 100],
                'dropout tes mcc': [best_test_perf['mcc']],
                'tes acc': [cur_tes_perf['acc'] * 100],
                'tes mcc': [cur_tes_perf['mcc']],
                'tes acc of y to log return' : [tes_matches_gt_acc * 100],
                'best benchmark tes total return': [tes_best_benchmark_model['total tes return']], 
                'best benchmark tes total profit per trade': [tes_best_benchmark_model['total tes profit per trade']],
                'best benchmark tes avg sharpe ratio': [tes_best_benchmark_model['tes sharpe ratio']],
                'best benchmark tes total log return': [tes_best_benchmark_model['total tes log return']],
                'best benchmark tes acc': [tes_best_benchmark_model['test accuracy']]                                                                                  
            }
        
        df_2 = pd.DataFrame(perf_dict_2)

    return df, df_2, val_mappings_df, tes_mappings_df

def get_signals(prob_arr, best_pred, best_gt, mappings,
                run, method, dataset, dropout, prob_measure,
                p, hinge, additional_metrics, ret_prob_df = None):
    mappings_pred_mask_arr = []
    mappings_gt_mask_arr = []
    mappings_arr = []

    #calculate signals by catering for uncertainty, filtering based on the threshold level
    for i, ma in enumerate(mappings):
        if ret_prob_df is not None:
            ticker_mapping = ret_prob_df[(ret_prob_df['ticker filename'] == ma['ticker_filename'])]      
            ticker_mapping = ticker_mapping.reset_index()        
            p = float(ticker_mapping['best prob'])
        m = copy.deepcopy(ma)
        m['run'] = run
        m['method'] = method
        m['dataset'] = dataset
        m['dropout'] = dropout
        m['prob_measure'] = prob_measure
        prob = prob_arr[i]
        gt = best_gt[i][0]
        m['prob'] = prob
        m['prob_filter'] = p
        pred = best_pred[i][0]
        m['day_pred'] = pred
        m['gt'] = gt

        perc = (prob/np.max(prob_arr)) * 100
        #perc = (prob/m['max_prob']) * 100
    

        if pred == 0:
            m['day_action_pred'] = -1
        else:
            m['day_action_pred'] = 1

        if abs(m['return']) > 100:
            ret = m['return']
        if perc > (100-(p*100)):
        #if prob > (1-p):
            m['day_action'] = 0
            mappings_pred_mask_arr.append(False)
            mappings_gt_mask_arr.append(False)
        elif pred == 0:
            m['day_action'] = -1
            mappings_pred_mask_arr.append(True)
            mappings_gt_mask_arr.append(True)
        elif pred == 1:
            m['day_action'] = 1
            mappings_pred_mask_arr.append(True)
            mappings_gt_mask_arr.append(True)

        gt_matches_return = True if (m['gt'] == 1 and m['return'] > 0) or (m['gt'] == 0 and m['return'] < 0) else False
        if gt_matches_return == True:
            m['matches_gt_day'] = 1
        else:
            m['matches_gt_day'] = 0
        
        mappings_arr.append(m)

    pred_mask = best_pred[mappings_pred_mask_arr]
    gt_mask = best_gt[mappings_gt_mask_arr]
    if pred_mask.shape[0] == 0:
        pred_mask = np.zeros((best_pred.shape[0], best_pred.shape[1]))
    if gt_mask.shape[0] == 0:
        gt_mask = np.zeros((best_gt.shape[0], best_gt.shape[1]))

    cur_perf = evaluate(pred_mask, gt_mask, hinge, additional_metrics=additional_metrics)
    return mappings_arr, cur_perf

def get_ticker_aggregate(tickers, mappings_df, best_benchmark_model,
          method, dataset, dropout, run, p, prob_measure, ret_prob_df = None):
    
    ret_dic = {'method': [],
                'dataset': [],
                'dropout': [],
                'run': [], 
                'prob': [],
                'prob measure': [],
                'ticker filename': [], 
                'total return': [], 
                'avg return': [], 
                'total log return': [], 
                'avg log return': [], 
                'profit per trade': [],
                'sharpe ratio': [], 
                'best benchmark log return': [], 
                'best benchmark profit per trade': [], 
                'best benchmark return': [],
                'best benchmark sharpe ratio': [], 
                'total trading days skipped': [], 
                'total trading days profitable': [], 
                'total trading days': [], 
                'total trading days averted loss by skipping': [], 
                'std log return before trading': [],
                'log return beats benchmark':[],
                'sharpe ratio beats benchmark':[]}
    for t in tickers:
        ticker_mapping = mappings_df[(mappings_df['ticker_filename'] == t)]
        ticker_mapping_without_skipped_days = ticker_mapping[(ticker_mapping['return_action'] != 0)]
        total_log_return = ticker_mapping_without_skipped_days['log_return_action'].sum()
        avg_log_return = ticker_mapping_without_skipped_days['log_return_action'].mean()
        total_return = ticker_mapping_without_skipped_days['return_action'].sum()
        avg_total_return = ticker_mapping_without_skipped_days['return_action'].mean()

        sharp_ratio = avg_total_return / ticker_mapping['return_action'].std() 
        std_log_return_before_action = ticker_mapping['return'].std()        

        total_trading_days_skipped = ticker_mapping[(ticker_mapping['day_action'] == 0)].shape[0]
        total_trading_days_successful = ticker_mapping[((ticker_mapping['log_return_action'] > 0) & (ticker_mapping['day_action'] > 0)) | ((ticker_mapping['log_return_action'] < 0) & (ticker_mapping['day_action'] < 0))].shape[0]
        total_trading_days = ticker_mapping.shape[0]
        total_trading_days_correctly_skipped = ticker_mapping[(ticker_mapping['day_action'] == 0) & (ticker_mapping['log_return_action_pred'] <= 0)].shape[0]
        
        best_benchmark_model_ticker = best_benchmark_model[(best_benchmark_model['ticker filename'] == t)]
        best_benchmark_log_return = best_benchmark_model_ticker['total log return'].iloc[0]
        best_benchmark_return = best_benchmark_model_ticker['total return'].iloc[0]
        best_benchmark_sharpe_ratio = best_benchmark_model_ticker['sharpe ratio'].iloc[0]
        best_benchmark_profit_per_trade = best_benchmark_model_ticker['profit per trade'].iloc[0]

        profit_per_trade = total_return / (total_trading_days - total_trading_days_skipped)

        if ret_prob_df is not None:
            ticker_mapping_prob = ret_prob_df[(ret_prob_df['ticker filename'] == t)]      
            ticker_mapping_prob = ticker_mapping_prob.reset_index()  
            p = float(ticker_mapping_prob['best prob'])
    
        ret_dic['method'].append(method)
        ret_dic['dataset'].append(dataset)
        ret_dic['dropout'].append(dropout)
        ret_dic['run'].append(run)  
        ret_dic['prob'].append(p)
        ret_dic['prob measure'].append(prob_measure)
        ret_dic['ticker filename'].append(t)
        ret_dic['total return'].append(total_return)
        ret_dic['avg return'].append(avg_total_return)
        ret_dic['total log return'].append(total_log_return)
        ret_dic['avg log return'].append(avg_log_return)
        ret_dic['profit per trade'].append(profit_per_trade)
        ret_dic['sharpe ratio'].append(sharp_ratio)
        ret_dic['best benchmark log return'].append(best_benchmark_log_return)
        ret_dic['best benchmark return'].append(best_benchmark_return)
        ret_dic['best benchmark profit per trade'].append(best_benchmark_profit_per_trade)
        ret_dic['best benchmark sharpe ratio'].append(best_benchmark_sharpe_ratio)
        ret_dic['total trading days skipped'].append(total_trading_days_skipped)
        ret_dic['total trading days profitable'].append(total_trading_days_successful)
        ret_dic['total trading days'].append(total_trading_days - total_trading_days_skipped)
        ret_dic['total trading days averted loss by skipping'].append(total_trading_days_correctly_skipped)
        ret_dic['std log return before trading'].append(std_log_return_before_action)
        ret_dic['log return beats benchmark'].append(True if total_log_return > best_benchmark_log_return else False)
        ret_dic['sharpe ratio beats benchmark'].append(True if sharp_ratio > best_benchmark_sharpe_ratio else False)

    ret_df = pd.DataFrame(ret_dic)
    return ret_df

def save_file(df, path):
    full_path = ''
    split_path = path.split('/')

    for ind, i in enumerate(split_path):
        if ind + 1 < len(split_path):
            full_path = full_path + i
            if not os.path.exists(full_path):
                os.mkdir(full_path)
            full_path = full_path + '/'

       
    df.to_csv(path, index = False)

def concat(df1, df2):
    if df1 is None:
        df1 = df2
    else:
        df1 = pd.concat([df1, df2])   

    return df1

def run_experiment_2_replication(predefined_args, args):      
    perf_df = None
    perf_df2 = None
    perf_df3 = None
    perf_df4 = None
    perf_ret_val_df = None
    perf_ret_tes_df = None
    for pre in predefined_args:
        args.path = pre['path']
        args.att = pre['att']
        args.seq = pre['seq']
        args.unit = pre['unit']
        args.alpha_l2 = pre['alpha_l2']
        args.fix_init = pre['fix_init']
        args.adv = pre['adv']
        args.reload = pre['reload']
        args.beta_adv = pre['beta_adv']
        args.epsilon_adv = pre['epsilon_adv']
        args.model_path = pre['model_path']
        method = pre['method']
        dataset = pre['dataset']
        if dataset == 'stocknet':
            tra_date = '2014-01-02'
            val_date = '2015-08-03'
            tes_date = '2015-10-01'
        elif dataset == 'kdd17':
            tra_date = '2007-01-03'
            val_date = '2015-01-02'
            tes_date = '2016-01-04'
        else:
            print('unexpected path: %s' % args.path)
            exit(0)

        parameters = {
            'seq': int(args.seq),
            'unit': int(args.unit),
            'alp': float(args.alpha_l2),
            'bet': float(args.beta_adv),
            'eps': float(args.epsilon_adv),
            'lr': float(args.learning_rate),
            'meth': method,
            'data': dataset,
            'act': args.action
        }
                

        pure_LSTM = AWLSTM(
            data_path=args.path,
            model_path=args.model_path,
            model_save_path=args.model_save_path,
            parameters=parameters,
            steps=args.step,
            epochs=args.epoch, batch_size=args.batch_size, gpu=args.gpu,
            tra_date=tra_date, val_date=val_date, tes_date=tes_date, att=args.att,
            hinge=args.hinge_lose, fix_init=args.fix_init, adv=args.adv,
            reload=args.reload,
            dropout_activation_function = None,
            load_mappings = True
        )

        save_path = args.model_save_path.replace('/model', '') + '/' + dataset
        mappings_save_path = save_path
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        if os.path.exists(mappings_save_path + '/val_mappings.pkl'):
            os.remove(mappings_save_path + '/val_mappings.pkl')

        if os.path.exists(mappings_save_path + '/tes_mappings.pkl'):
            os.remove(mappings_save_path + '/tes_mappings.pkl')

        val_mappings = copy.copy(pure_LSTM.val_mappings)
        with open(mappings_save_path + '/val_mappings.pkl', 'wb') as val_mappings_file:
            pickle.dump(val_mappings, val_mappings_file)

        tes_mappings = copy.copy(pure_LSTM.tes_mappings)
        with open(mappings_save_path + '/tes_mappings.pkl', 'wb') as tes_mappings_file:
            pickle.dump(tes_mappings, tes_mappings_file)

        runs = 5
        runs_arr = [*range(runs + 1)][1:]    

        for r in runs_arr:

            val_mappings_arr = []
            tes_mappings_arr = []
            best_valid_perf, best_test_perf = pure_LSTM.train(return_perf=True, return_pred=False)
            best_valid_pred = best_valid_perf['pred']
            best_test_pred = best_test_perf['pred']
            
            for i, ma in enumerate(val_mappings):
                m = copy.deepcopy(ma)
                m['run'] = r
                m['method'] = method
                m['dataset'] = dataset
                pred = best_valid_perf['pred'][i][0] 
                gt = best_valid_perf['gt'][i][0]
                val_mapping_return = m['return']
                if pred == gt and gt == 1 and val_mapping_return < 0:
                    g = 1
                elif pred == gt and gt == 0 and val_mapping_return > 0:
                    g = 1
                elif val_mapping_return == 0:
                    g = 0

                if pred == 0:
                    m['day_action'] = -1
                elif pred == 1:
                    m['day_action'] = 1
    
                val_mappings_arr.append(m)

            for i, ma in enumerate(tes_mappings):
                m = copy.deepcopy(ma)
                m['run'] = r
                m['method'] = method
                m['dataset'] = dataset
                pred = best_test_perf['pred'][i][0]
                if pred == 0:
                    m['day_action'] = -1
                elif pred == 1:
                    m['day_action'] = 1
                    
                tes_mappings_arr.append(m)
        
            val_mappings_df = pd.DataFrame(val_mappings_arr)
            tes_mappings_df = pd.DataFrame(tes_mappings_arr)

            val_mappings_df['log_return_action'] = val_mappings_df['log_return'] * val_mappings_df['day_action']
            val_mappings_df['return_action'] = val_mappings_df['return'] * val_mappings_df['day_action']
            val_pre_returns = val_mappings_df['return_action'].sum()
            val_total_investments = val_mappings_df['prev_adj_close'].abs().sum()
            val_pre_log_returns = val_mappings_df['log_return_action'].sum()

            tes_mappings_df['log_return_action'] = tes_mappings_df['log_return'] * tes_mappings_df['day_action']
            tes_mappings_df['return_action'] = tes_mappings_df['return'] * tes_mappings_df['day_action']
            tes_pre_returns = tes_mappings_df['return_action'].sum()
            tes_total_investments = tes_mappings_df['prev_adj_close'].abs().sum()
            tes_pre_log_returns = tes_mappings_df['log_return_action'].sum()

            # val_pre_returns_avg = val_mappings_df['log_return_action'].mean()
            # tes_pre_returns_avg = tes_mappings_df['log_return_action'].mean()
            val_pre_returns_avg = val_mappings_df['return_action'].mean()
            tes_pre_returns_avg = tes_mappings_df['return_action'].mean()
        
            # val_sharp_ratio = val_pre_returns_avg / val_mappings_df['log_return_action'].std()
            # tes_sharp_ratio = tes_pre_returns_avg / tes_mappings_df['log_return_action'].std()
        
            val_sharp_ratio = val_pre_returns_avg / val_mappings_df['return_action'].std()
            tes_sharp_ratio = tes_pre_returns_avg / tes_mappings_df['return_action'].std()

            val_total_trading_days_skipped = val_mappings_df[(val_mappings_df['day_action'] == 0)].shape[0]
            tes_total_trading_days_skipped = tes_mappings_df[(tes_mappings_df['day_action'] == 0)].shape[0]
            # val_total_trading_days_successful = val_mappings_df[(val_mappings_df['log_return_action'] > 0)].shape[0]
            # tes_total_trading_days_successful = tes_mappings_df[(tes_mappings_df['log_return_action'] > 0)].shape[0]
            val_total_trading_days_successful = val_mappings_df[(val_mappings_df['return_action'] > 0)].shape[0]
            tes_total_trading_days_successful = tes_mappings_df[(tes_mappings_df['return_action'] > 0)].shape[0]
            
            total_val_trading_days = val_mappings_df.shape[0]
            total_tes_trading_days = tes_mappings_df.shape[0]
        
            perf_dict = {
                    'method': [method],
                    'dataset': [dataset],
                    'val accuracy': [best_valid_perf['acc'] * 100],  
                    'val mcc': [best_valid_perf['mcc']],  
                    'total val trading days skipped': [val_total_trading_days_skipped],
                    'total val trading days successful': [val_total_trading_days_successful],
                    'total val trading days': [total_val_trading_days-val_total_trading_days_skipped],
                    'total val return': [val_pre_returns],
                    'total val log return': [val_pre_log_returns],
                    'total val profit per trade': [val_pre_returns/(total_val_trading_days-val_total_trading_days_skipped)],
                    'val total investment': [val_total_investments],
                    'avg val return': [val_pre_returns_avg],
                    'val sharpe ratio': [val_sharp_ratio],
                    'test accuracy': [best_test_perf['acc'] * 100],  
                    'test mcc': [best_test_perf['mcc']],  
                    'total tes trading days skipped': [tes_total_trading_days_skipped],
                    'total tes trading days successful': [tes_total_trading_days_successful],
                    'total tes trading days': [total_tes_trading_days-tes_total_trading_days_skipped],
                    'total tes profit per trade': [tes_pre_returns/(total_tes_trading_days-tes_total_trading_days_skipped)],
                    'total tes return': [tes_pre_returns],
                    'total tes log return': [tes_pre_log_returns],
                    'tes total investment': [tes_total_investments],
                    'avg tes return': [tes_pre_returns_avg],
                    'tes sharpe ratio': [tes_sharp_ratio],
                    'run': [r]
                }

            df = pd.DataFrame(perf_dict)
            if perf_df is None:
                perf_df = df
            else:
                perf_df = pd.concat([perf_df, df])            

            if not os.path.exists('experiment2'):
                os.mkdir('experiment2')
            if not os.path.exists('experiment2/replication'):
                os.mkdir('experiment2/replication')
            perf_df.to_csv('experiment2/replication/replication_pre_returns_results.csv', index = False)
            #dfi.export(perf_df,"experiment2/replication_pre_returns_results.png")

            tickers = set(list(map(lambda x: x['ticker_filename'], val_mappings_arr)))

            ret_val_dic = {'method': [], 'dataset': [], 'run': [], 'ticker filename': [], 'total return': [], 'profit per trade': [], 'total log return': [], 'total investment':[], 'avg return': [], 'avg log return': [], 'sharpe ratio': [], 'total trading days skipped': [], 'total trading days successful': [], 'total trading days': []}
            for t in tickers:
                ticker_mapping = val_mappings_df[(val_mappings_df['ticker_filename'] == t)]
                total_log_return = ticker_mapping['log_return_action'].sum()
                avg_log_return = ticker_mapping['log_return_action'].mean()  
                total_return = ticker_mapping['return_action'].sum()
                avg_return = ticker_mapping['return_action'].mean()    
                total_return_inv = ticker_mapping['prev_adj_close'].abs().sum()   

                #sharp_ratio = avg_log_return / ticker_mapping['log_return_action'].std() 
                sharp_ratio = avg_return / ticker_mapping['return_action'].std() 
                total_trading_days_skipped = ticker_mapping[(ticker_mapping['day_action'] == 0)].shape[0]
                #total_trading_days_successful = ticker_mapping[((ticker_mapping['log_return_action'] > 0) & (ticker_mapping['day_action'] > 0)) | ((ticker_mapping['log_return_action'] < 0) & (ticker_mapping['day_action'] < 0))].shape[0]
                total_trading_days_successful = ticker_mapping[((ticker_mapping['return_action'] > 0) & (ticker_mapping['day_action'] > 0)) | ((ticker_mapping['return_action'] < 0) & (ticker_mapping['day_action'] < 0))].shape[0]
                total_trading_days = ticker_mapping.shape[0]

                ret_val_dic['method'].append(method)
                ret_val_dic['dataset'].append(dataset)
                ret_val_dic['run'].append(r)  
                ret_val_dic['ticker filename'].append(t)
                ret_val_dic['total return'].append(total_return)
                ret_val_dic['total log return'].append(total_log_return)
                ret_val_dic['profit per trade'].append(total_return / (total_trading_days-total_trading_days_skipped))
                ret_val_dic['avg return'].append(avg_return)
                ret_val_dic['avg log return'].append(avg_log_return)
                ret_val_dic['total investment'].append(total_return_inv)
                ret_val_dic['sharpe ratio'].append(sharp_ratio)
                ret_val_dic['total trading days skipped'].append(total_trading_days_skipped)
                ret_val_dic['total trading days successful'].append(total_trading_days_successful)
                ret_val_dic['total trading days'].append(total_trading_days)

            ret_val_df = pd.DataFrame(ret_val_dic)
            if perf_ret_val_df is None:
                perf_ret_val_df = ret_val_df
            else:
                perf_ret_val_df = pd.concat([perf_ret_val_df, ret_val_df])      
            if not os.path.exists('experiment2'):
                os.mkdir('experiment2')
            if not os.path.exists('experiment2/replication'):
                os.mkdir('experiment2/replication')
            perf_ret_val_df.to_csv('experiment2/replication/replication_pre_val_ticker_returns_results.csv', index = False)


            tickers = set(list(map(lambda x: x['ticker_filename'], tes_mappings_arr)))
            ret_tes_dic = {'method': [], 'dataset': [], 'run': [], 'ticker filename': [], 'total return': [], 'profit per trade': [], 'total log return': [], 'avg return': [], 'avg log return': [], 'total investment': [], 'sharpe ratio': [], 'total trading days skipped': [], 'total trading days successful': [], 'total trading days': []}

            for t in tickers:
                ticker_mapping = tes_mappings_df[(tes_mappings_df['ticker_filename'] == t)]
                total_log_return = ticker_mapping['log_return_action'].sum()
                avg_log_return = ticker_mapping['log_return_action'].mean() 
                total_return = ticker_mapping['return_action'].sum()
                total_return_inv = ticker_mapping['prev_adj_close'].abs().sum()
                avg_return = ticker_mapping['return_action'].mean()       
                sharp_ratio = avg_return / ticker_mapping['return_action'].std()  
                # sharp_ratio = avg_return / ticker_mapping['log_return_action'].std() 
                total_trading_days_skipped = ticker_mapping[(ticker_mapping['day_action'] == 0)].shape[0]
                # total_trading_days_successful = ticker_mapping[((ticker_mapping['log_return_action'] > 0) & (ticker_mapping['day_action'] > 0)) | ((ticker_mapping['log_return_action'] < 0) & (ticker_mapping['day_action'] < 0))].shape[0]
                total_trading_days_successful = ticker_mapping[((ticker_mapping['return_action'] > 0) & (ticker_mapping['day_action'] > 0)) | ((ticker_mapping['return_action'] < 0) & (ticker_mapping['day_action'] < 0))].shape[0]

                total_trading_days = ticker_mapping.shape[0]

                ret_tes_dic['method'].append(method)
                ret_tes_dic['dataset'].append(dataset)
                ret_tes_dic['run'].append(r)  
                ret_tes_dic['ticker filename'].append(t)
                ret_tes_dic['total return'].append(total_return)
                ret_tes_dic['total log return'].append(total_log_return)
                ret_tes_dic['profit per trade'].append(total_return / (total_trading_days-total_trading_days_skipped))
                ret_tes_dic['avg return'].append(avg_return)
                ret_tes_dic['avg log return'].append(avg_log_return)
                ret_tes_dic['total investment'].append(total_return_inv)
                ret_tes_dic['sharpe ratio'].append(sharp_ratio)
                ret_tes_dic['total trading days skipped'].append(total_trading_days_skipped)
                ret_tes_dic['total trading days successful'].append(total_trading_days_successful)
                ret_tes_dic['total trading days'].append(total_trading_days)

            ret_tes_df = pd.DataFrame(ret_tes_dic)
            if perf_ret_tes_df is None:
                perf_ret_tes_df = ret_tes_df
            else:
                perf_ret_tes_df = pd.concat([perf_ret_tes_df, ret_tes_df])      
            if not os.path.exists('experiment2'):
                os.mkdir('experiment2')
            if not os.path.exists('experiment2/replication'):
                os.mkdir('experiment2/replication')
            perf_ret_tes_df.to_csv('experiment2/replication/replication_pre_tes_ticker_returns_results.csv', index = False)

            df_3 = pd.DataFrame(val_mappings_arr)

            if perf_df3 is None:
                perf_df3 = df_3
            else:
                perf_df3 = pd.concat([perf_df3, df_3])         

            if not os.path.exists('experiment2'):
                os.mkdir('experiment2')
            if not os.path.exists('experiment2/replication'):
                os.mkdir('experiment2/replication')
            perf_df3.to_csv('experiment2/replication/replication_val_mapping_results.csv', index = False)
            #dfi.export(perf_df3,"experiment2/replication_val_mapping_results.png")

            df_4 = pd.DataFrame(tes_mappings_arr)

            if perf_df4 is None:
                perf_df4 = df_4
            else:
                perf_df4 = pd.concat([perf_df4, df_4])         

            if not os.path.exists('experiment2'):
                os.mkdir('experiment2')
            if not os.path.exists('experiment2/replication'):
                os.mkdir('experiment2/replication')
            perf_df4.to_csv('experiment2/replication/replication_tes_mapping_results.csv', index = False)
            #dfi.export(perf_df4,"experiment2/replication_tes_mapping_results.png")

        avg_total_val_pre_returns = np.average(perf_df[(perf_df['dataset'] == dataset)]['total val return'].to_numpy())
        avg_total_tes_pre_returns = np.average(perf_df[(perf_df['dataset'] == dataset) ]['total tes return'].to_numpy())
        avg_val_pre_returns = np.average(perf_df[(perf_df['dataset'] == dataset)]['avg val return'].to_numpy())
        avg_tes_pre_returns = np.average(perf_df[(perf_df['dataset'] == dataset)]['avg tes return'].to_numpy())

        avg_sharp_ratio_val = np.average(perf_df[(perf_df['dataset'] == dataset)]['val sharpe ratio'].to_numpy())
        avg_sharp_ratio_tes = np.average(perf_df[(perf_df['dataset'] == dataset)]['tes sharpe ratio'].to_numpy())

        perf_dict_2 = {
                    'method': [method],
                    'dataset': [dataset],
                    'avg val accuracy': [np.average(perf_df[(perf_df['dataset'] == dataset)]['val accuracy'].to_numpy())],
                    'avg tes accuracy': [np.average(perf_df[(perf_df['dataset'] == dataset)]['test accuracy'].to_numpy())],
                    'avg val mcc': [np.average(perf_df[(perf_df['dataset'] == dataset)]['val mcc'].to_numpy())],
                    'avg tes mcc': [np.average(perf_df[(perf_df['dataset'] == dataset)]['test mcc'].to_numpy())],      
                    'avg total val investment': [np.average(perf_df[(perf_df['dataset'] == dataset)]['val total investment'].to_numpy())],
                    'avg total tes investment': [np.average(perf_df[(perf_df['dataset'] == dataset)]['tes total investment'].to_numpy())],
                    'avg val profit per trade': [np.average(perf_df[(perf_df['dataset'] == dataset)]['total val profit per trade'].to_numpy())],
                    'avg tes profit per trade': [np.average(perf_df[(perf_df['dataset'] == dataset)]['total tes profit per trade'].to_numpy())],
                    'std val profit per trade': [np.std(perf_df[(perf_df['dataset'] == dataset)]['total val profit per trade'].to_numpy())],
                    'std tes profit per trade': [np.std(perf_df[(perf_df['dataset'] == dataset)]['total tes profit per trade'].to_numpy())],
                    'avg total val predicted return': [avg_total_val_pre_returns],
                    'avg total tes predicted return': [avg_total_tes_pre_returns],
                    'std total val predicted return': [np.std(perf_df[(perf_df['dataset'] == dataset)]['total val return'].to_numpy())],
                    'std total tes predicted return': [np.std(perf_df[(perf_df['dataset'] == dataset)]['total tes return'].to_numpy())],
                    'avg val predicted return': [avg_val_pre_returns],
                    'avg tes predicted return': [avg_tes_pre_returns],
                    'avg val sharpe ratio': [avg_sharp_ratio_val],
                    'avg tes sharpe ratio': [avg_sharp_ratio_tes],
                    'avg val log return': [np.average(perf_df[(perf_df['dataset'] == dataset)]['total val log return'].to_numpy())],
                    'avg tes log return': [np.average(perf_df[(perf_df['dataset'] == dataset)]['total tes log return'].to_numpy())],           
                }
        df_2 = pd.DataFrame(perf_dict_2)

        if perf_df2 is None:
            perf_df2 = df_2
        else:
            perf_df2 = pd.concat([perf_df2, df_2])         

        if not os.path.exists('experiment2'):
            os.mkdir('experiment2')
        if not os.path.exists('experiment2/replication'):
            os.mkdir('experiment2/replication')
        perf_df2.to_csv('experiment2/replication/replication_pre_returns_grouped_results.csv', index = False)

def run_experiment_2_dropout():
    prob_arr = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
    #prob_arr = [1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.5, 0]
    perf_df_v = None
    perf_df_t = None
    perf_df2 = None
    perf_df3 = None
    perf_df4 = None
    perf_ret_val_df = None
    perf_ret_tes_df = None
    perf_val_tuned_df = None
    perf_tes_tuned_df = None
    perf_ret_val_prob_df = None
    perf_ret_val_tuned_df = None
    perf_ret_tes_tuned_df = None

    benchmark_df = pd.read_csv('./experiment2/replication/replication_pre_returns_results.csv')
    benchmark_val_df = pd.read_csv('./experiment2/replication/replication_pre_val_ticker_returns_results.csv')
    benchmark_tes_df = pd.read_csv('./experiment2/replication/replication_pre_tes_ticker_returns_results.csv')
    benchmark_dp_df = pd.read_csv('./experiment1/dropout/perf_dropout_results.csv')                
    benchmark_dp_uncertainty_df = pd.read_csv('./experiment1/dropout_uncertainty/perf_dropout_results.csv')      
               

    if os.path.exists('experiment2/dropout/dropout_val_mapping_results.csv'):
        os.remove('experiment2/dropout/dropout_val_mapping_results.csv')

    if os.path.exists('experiment2/dropout/dropout_tes_mapping_results.csv'):
        os.remove('experiment2/dropout/dropout_tes_mapping_results.csv')

    for index, row in benchmark_dp_df.iterrows():
        method = row['method']
        dataset = row['dataset']
        dropout = row['dropout']
        run = row['run']

        #get best experiment 1 dropout run by validation accuracy
        best_benchmark_dp_model = benchmark_dp_df[(benchmark_dp_df['dataset'] == dataset) & (benchmark_dp_df['method'] == method) & (benchmark_dp_df['dropout'] == dropout)]
        best_benchmark_dp_model = best_benchmark_dp_model.reset_index()
        val_best_ind = best_benchmark_dp_model['valid acc'].idxmax()
        val_best_benchmark_model = best_benchmark_dp_model.iloc[[val_best_ind]]
        val_best_benchmark_model = val_best_benchmark_model.reset_index()
        val_best_ind = val_best_benchmark_model['valid mcc'].idxmax()
        val_best_benchmark_model = val_best_benchmark_model.iloc[val_best_ind]

        #dropout = val_best_benchmark_model['dropout']

        benchmark_dp_uncertainty_df_model = benchmark_dp_uncertainty_df[(benchmark_dp_uncertainty_df['dataset'] == dataset) & (benchmark_dp_uncertainty_df['method'] == method) & (benchmark_dp_uncertainty_df['dropout'] == dropout)]
        benchmark_dp_uncertainty_df_model = benchmark_dp_uncertainty_df_model.reset_index()
        val_best_ind = benchmark_dp_uncertainty_df_model['valid acc'].idxmax()
        val_best_benchmark_model = benchmark_dp_uncertainty_df_model.iloc[[val_best_ind]]
        # val_best_benchmark_model = val_best_benchmark_model.reset_index()
        # val_best_ind = val_best_benchmark_model['valid acc'].idxmax()
        # val_best_benchmark_model = val_best_benchmark_model.iloc[val_best_ind]

        #continue because we want to do the experiment only on the best dropout
        # if dropout != row ['dropout']:
        #     continue
        #get best experiment 2 replication run
        best_benchmark_model = benchmark_df[(benchmark_df['dataset'] == dataset) & (benchmark_df['method'] == method)]
        best_benchmark_model = best_benchmark_model.reset_index()
        val_best_ind = best_benchmark_model['total val log return'].idxmax()
        #val_best_ind = best_benchmark_model['val accuracy'].idxmax()
        val_best_benchmark_model = best_benchmark_model.iloc[val_best_ind]
        val_best_benchmark_model_run = int(val_best_benchmark_model['run'])

        #get best experiment 2 replication run
        best_benchmark_model = best_benchmark_model.reset_index()
        tes_best_ind = best_benchmark_model['total val log return'].idxmax()
        #tes_best_ind = best_benchmark_model['val accuracy'].idxmax()
        tes_best_benchmark_model = best_benchmark_model.iloc[tes_best_ind]
        tes_best_benchmark_model_run = int(tes_best_benchmark_model['run'])

        #get the validation and test results on the ticker level for the according best experiment 2 replication run
        best_benchmark_val_model = benchmark_val_df[(benchmark_val_df['dataset'] == dataset) & (benchmark_val_df['method'] == method) & (benchmark_val_df['run'].astype(int) == val_best_benchmark_model_run)]
        best_benchmark_tes_model = benchmark_tes_df[(benchmark_tes_df['dataset'] == dataset) & (benchmark_tes_df['method'] == method) & (benchmark_tes_df['run'].astype(int) == tes_best_benchmark_model_run)] 
        
        run_save_path = row['run save path']
        
        #load the validation and test mappings
        mappings_save_path = './tmp/' + dataset
        with open(mappings_save_path + '/val_mappings.pkl', 'rb') as val_mappings_file:
            val_mappings = pickle.load(val_mappings_file)
        with open(mappings_save_path + '/tes_mappings.pkl', 'rb') as tes_mappings_file:
            tes_mappings = pickle.load(tes_mappings_file)

        #load validation and test performance files
        with open(run_save_path + '/best_test_perf.pkl', "rb") as input_file:
            best_test_perf = pickle.load(input_file)
        with open(run_save_path + '/best_valid_perf.pkl', "rb") as input_file:
            best_valid_perf = pickle.load(input_file)
        
        for mi, m in enumerate(best_valid_perf['prob_arr']):
            val_pre_prob = best_valid_perf['prob_arr'][mi]['val']
            tes_pre_prob = best_test_perf['prob_arr'][mi]['val']
            print('Processing Dropout: ' + str(dropout) + ', Run: ' + str(run) + ', Method: ' + method + ', Dataset: ' + dataset + ', Prob Measure: ' + best_valid_perf['prob_arr'][mi]['measure'])

            #val_mappings = get_max_prob_mappings(val_pre_prob, val_mappings)
            #tes_mappings = get_max_prob_mappings(tes_pre_prob, tes_mappings)

            prob_measure = best_valid_perf['prob_arr'][mi]['measure']

            val_mappings_tuned_arr = []

            #iterate at each of the threshold levels
            for p in prob_arr:
            
                #calculate signals by catering for uncertainty, filtering based on the threshold level (validation)
                val_mappings_arr, cur_valid_perf = get_signals(val_pre_prob, best_valid_perf['pred'], best_valid_perf['gt'], val_mappings,
                                                 
                                                run, method, dataset, dropout, prob_measure,
                                                p, best_valid_perf['hinge'], best_valid_perf['additional_metrics'])
                
                #calculate signals by catering for uncertainty, filtering based on the threshold level (test)
                tes_mappings_arr, cur_tes_perf = get_signals(tes_pre_prob, best_test_perf['pred'], best_test_perf['gt'], tes_mappings,
                                    run, method, dataset, dropout, prob_measure,
                                    p, best_test_perf['hinge'], best_test_perf['additional_metrics'])


                val_mappings_df = pd.DataFrame(val_mappings_arr)          
                tes_mappings_df = pd.DataFrame(tes_mappings_arr)

                #Calculate the returns and aggregate data for the model
                df_v, df_t, val_mappings_df,  tes_mappings_df = get_run_aggregate(val_mappings_df, tes_mappings_df,
                                                                       best_valid_perf, best_test_perf,
                                                                       cur_valid_perf, cur_tes_perf,
                                                                       val_best_benchmark_model, tes_best_benchmark_model, method,
                                                                       dataset, dropout, run, p, prob_measure)
                perf_df_v = concat(perf_df_v, df_v)
                save_file(perf_df_v, 'experiment2/dropout/run/dropout_pre_returns_val_results.csv')   

                perf_df_t = concat(perf_df_t, df_t)
                save_file(perf_df_t, 'experiment2/dropout/run/dropout_pre_returns_tes_results.csv')   

                #Aggregate data per ticker (validation)
                val_tickers = set(list(map(lambda x: x['ticker_filename'], val_mappings_arr)))
                ret_val_df = get_ticker_aggregate(val_tickers, val_mappings_df, best_benchmark_val_model,
                                                  method, dataset, dropout, run, p, prob_measure)
                perf_ret_val_df = concat(perf_ret_val_df, ret_val_df)
                save_file(perf_ret_val_df, 'experiment2/dropout/ticker/dropout_pre_val_ticker_returns_results.csv')

                #Aggregate data per ticker (test)
                tes_tickers = set(list(map(lambda x: x['ticker_filename'], tes_mappings_arr)))
                ret_tes_df = get_ticker_aggregate(tes_tickers, tes_mappings_df, best_benchmark_tes_model,
                                                  method, dataset, dropout, run, p, prob_measure)
                perf_ret_tes_df = concat(perf_ret_tes_df, ret_tes_df)
                save_file(perf_ret_tes_df, 'experiment2/dropout/ticker/dropout_pre_tes_ticker_returns_results.csv')

                #Results for each trading day (validation)
                perf_df3 = concat(perf_df3, val_mappings_df)
                save_file(perf_df3, 'experiment2/dropout/day/dropout_val_mapping_results.csv')
  
                #Results for each trading day (test)
                perf_df4 = concat(perf_df4, tes_mappings_df)
                save_file(perf_df4, 'experiment2/dropout/day/dropout_tes_mapping_results.csv')

                for t in val_tickers:
                    val_mappings_df_ticker = val_mappings_df[(val_mappings_df['ticker_filename'] == t)]
                    save_file(val_mappings_df_ticker, 'experiment2/dropout/day/dropout_val_mappings_results/' + dataset + '/' + prob_measure + '/' + str(p) + '/' + t)

                for t in tes_tickers:
                    tes_mappings_df_ticker = tes_mappings_df[(tes_mappings_df['ticker_filename'] == t)]
                    save_file(tes_mappings_df_ticker, 'experiment2/dropout/day/dropout_tes_mappings_results/' + dataset + '/' + prob_measure + '/' + str(p) + '/' + t)

            #aggregate data before tuning
            for s in prob_arr:
                avg_total_val_pre_returns = np.average(perf_df_v[(perf_df_v['method'] == method) & (perf_df_v['dataset'] == dataset) & (perf_df_v['dropout'] == dropout) & (perf_df_v['prob'] == s) & (perf_df_v['prob measure'] == prob_measure)]['total val return'].to_numpy())
                avg_total_tes_pre_returns = np.average(perf_df_t[(perf_df_t['method'] == method) & (perf_df_t['dataset'] == dataset) & (perf_df_t['dropout'] == dropout) & (perf_df_t['prob'] == s) & (perf_df_t['prob measure'] == prob_measure)]['total tes return'].to_numpy())
                avg_val_pre_returns = np.average(perf_df_v[(perf_df_v['method'] == method) & (perf_df_v['dataset'] == dataset) & (perf_df_v['dropout'] == dropout) & (perf_df_v['prob'] == s) & (perf_df_v['prob measure'] == prob_measure)]['avg val return'].to_numpy())
                avg_tes_pre_returns = np.average(perf_df_t[(perf_df_t['method'] == method) & (perf_df_t['dataset'] == dataset) & (perf_df_t['dropout'] == dropout) & (perf_df_t['prob'] == s) & (perf_df_t['prob measure'] == prob_measure)]['avg tes return'].to_numpy())

                avg_sharp_ratio_val = np.average(perf_df_v[(perf_df_v['method'] == method) & (perf_df_v['dataset'] == dataset) & (perf_df_v['dropout'] == dropout) & (perf_df_v['prob'] == s) & (perf_df_v['prob measure'] == prob_measure)]['val sharpe ratio'].to_numpy())
                avg_sharp_ratio_tes = np.average(perf_df_t[(perf_df_t['method'] == method) & (perf_df_t['dataset'] == dataset)  & (perf_df_t['dropout'] == dropout) & (perf_df_t['prob'] == s) & (perf_df_t['prob measure'] == prob_measure)]['tes sharpe ratio'].to_numpy())

                perf_dict_2 = {
                            'method': [method],
                            'dataset': [dataset],
                            'dropout': [dropout],
                            'avg total val predicted return': [avg_total_val_pre_returns],
                            'avg total tes predicted return': [avg_total_tes_pre_returns],
                            'avg val predicted return': [avg_val_pre_returns],
                            'avg tes predicted return': [avg_tes_pre_returns],
                            'avg val sharpe ratio': [avg_sharp_ratio_val],
                            'avg tes sharpe ratio': [avg_sharp_ratio_tes],
                            'prob': [s],
                            'prob measure' : [prob_measure]           
                        }
                df_2 = pd.DataFrame(perf_dict_2)
                perf_df2 = concat(perf_df2, df_2)
                save_file(perf_df2, 'experiment2/dropout/model/dropout_pre_returns_grouped_results.csv')

            #Find best prob for each ticker according to highest validation log return
            ret_val_prob_dic = {'method': [], 'dataset': [], 'dropout': [], 'run': [], 'best prob': [], 'prob measure': [], 'ticker filename': [], 'total val log return': []}
            for t in val_tickers:
                ticker_mapping = perf_ret_val_df[(perf_ret_val_df['ticker filename'] == t) & (perf_ret_val_df['method'] == method) & (perf_ret_val_df['dataset'] == dataset) & (perf_ret_val_df['dropout'] == dropout) & (perf_ret_val_df['prob measure'] == prob_measure)]                        
                ticker_mapping = ticker_mapping.reset_index()
                val_best_ind = ticker_mapping['total log return'].idxmax()
                val_best_prob = ticker_mapping.iloc[val_best_ind]['prob']
                val_best_log_return = ticker_mapping.iloc[val_best_ind]['total log return']

                ret_val_prob_dic['method'].append(method)
                ret_val_prob_dic['dataset'].append(dataset)
                ret_val_prob_dic['dropout'].append(dropout)
                ret_val_prob_dic['run'].append(run)  
                ret_val_prob_dic['best prob'].append(val_best_prob)
                ret_val_prob_dic['prob measure'].append(prob_measure)
                ret_val_prob_dic['ticker filename'].append(t)
                ret_val_prob_dic['total val log return'].append(val_best_log_return)

            ret_val_prob_df = pd.DataFrame(ret_val_prob_dic)
            perf_ret_val_prob_df = concat(perf_ret_val_prob_df, ret_val_prob_df)
            save_file(perf_ret_val_prob_df, 'experiment2/dropout/dropout_pre_val_prob_ticker_returns_results.csv')

            #calculate signals by catering for uncertainty, filtering based on the threshold level per ticker (validation)
            perf_val_tuned_df, perf_ret_val_tuned_df = tune_results(val_pre_prob, best_valid_perf, 
            val_mappings, run, method, dataset, dropout, prob_measure,
            ret_val_prob_df,
            True, val_best_benchmark_model, best_benchmark_val_model, perf_val_tuned_df,
            val_tickers, perf_ret_val_tuned_df)

            #calculate signals by catering for uncertainty, filtering based on the threshold level per ticker (test)
            perf_tes_tuned_df, perf_ret_tes_tuned_df = tune_results(tes_pre_prob, best_test_perf, 
            tes_mappings, run, method, dataset, dropout, prob_measure,
            ret_val_prob_df,
            False, tes_best_benchmark_model, best_benchmark_tes_model, perf_tes_tuned_df,
            tes_tickers, perf_ret_tes_tuned_df)


def get_max_prob_mappings(prob_arr, mappings):
    # prob_arr_reshape = np.reshape(prob_arr, (prob_arr.shape[1]))
    prob_df = pd.DataFrame(prob_arr, columns=['prob_array'])
    mapping_df = pd.DataFrame(mappings)
    prob_df = prob_df.merge(mapping_df, left_index=True, right_index=True)
    group_by_df = prob_df.groupby('ticker_filename').agg({'prob_array': 'max'})
    mappings_merge = mapping_df.merge(group_by_df, 
    left_on='ticker_filename', \
    right_on='ticker_filename', \
    how='inner', sort=False).sort_values(by='ins_ind', ascending=True).reset_index()

    mapping_df['max_prob'] = mappings_merge['prob_array']
    return mapping_df.to_dict(orient='records')

def tune_results(pre_prob, best_perf, 
mappings, run, method, dataset, dropout, prob_measure, ret_prob_df,

is_validation_set, best_benchmark_run_model, best_benchmark_ticker_model, perf_tuned_df,

tickers, ret_tuned_df):
    mappings_tuned_arr, cur_valid_perf = get_signals(pre_prob, best_perf['pred'], best_perf['gt'], 
    mappings, run, method, dataset, dropout, prob_measure, None, best_perf['hinge'], 
    best_perf['additional_metrics'], ret_prob_df)
    
    mappings_tuned_df = pd.DataFrame(mappings_tuned_arr)        

    if is_validation_set:   
        tuned_results_path = 'experiment2/dropout/run/dropout_pre_val_tuned_returns_results.csv'  
        ticker_results_path = 'experiment2/dropout/ticker/dropout_pre_val_prob_ticker_returns_results.csv' 
        df_tuned_v, df_tuned_t, mappings_tuned_df, placeholder = get_run_aggregate(
        mappings_tuned_df, None,
        best_perf, None,
        cur_valid_perf, None,
        best_benchmark_run_model, None, method,
        dataset, dropout, run, None, prob_measure)
        perf_tuned_df = concat(perf_tuned_df, df_tuned_v)
    else:
        tuned_results_path = 'experiment2/dropout/run/dropout_pre_tes_tuned_returns_results.csv'  
        ticker_results_path = 'experiment2/dropout/ticker/dropout_pre_tes_prob_ticker_returns_results.csv' 
        df_tuned_v, df_tuned_t, placeholder, mappings_tuned_df = get_run_aggregate(
        None, mappings_tuned_df,
        None, best_perf,
        None, cur_valid_perf,
        None, best_benchmark_run_model, method,
        dataset, dropout, run, None, prob_measure)
            
        perf_tuned_df = concat(perf_tuned_df, df_tuned_t)

    ret_tuned_df = get_ticker_aggregate(tickers, mappings_tuned_df, best_benchmark_ticker_model,
                                                method, dataset, dropout, run, None, prob_measure, ret_prob_df)
    perf_ret_tuned_df = concat(ret_tuned_df, ret_tuned_df)

    save_file(perf_tuned_df, tuned_results_path)
    save_file(perf_ret_tuned_df, ticker_results_path)

    return perf_tuned_df, perf_ret_tuned_df
                