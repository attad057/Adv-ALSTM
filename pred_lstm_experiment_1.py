from pred_lstm import AWLSTM, evaluate             
import pandas as pd
import os
import numpy as np
import dataframe_image as dfi
import pickle  

def run_experiment_1_dropout(predefined_args, args):
    perf_df = None
    perf_df2 = None

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
        args.model_save_path = pre['model_save_path']
        args.method = pre['method']
        args.dataset = pre['dataset']
        args.batch_size = pre['batch_size']
        if args.dataset == 'stocknet':
            args.tra_date = '2014-01-02'
            args.val_date = '2015-08-03'
            args.tes_date = '2015-10-01'
        elif args.dataset == 'kdd17':
            args.tra_date = '2007-01-03'
            args.val_date = '2015-01-02'
            args.tes_date = '2016-01-04'
        else:
            print('unexpected path: %s' % args.path)
            exit(0)

        for i, s in enumerate(args.state_keep_prob_arr):
            args.state_keep_prob = s
            parameters = {
                'seq': int(args.seq),
                'unit': int(args.unit),
                'alp': float(args.alpha_l2),
                'bet': float(args.beta_adv),
                'eps': float(args.epsilon_adv),
                'lr': float(args.learning_rate),
                'meth': args.method,
                'data': args.dataset,
                'act': args.action,
                'state_keep_prob': float(args.state_keep_prob),
                'batch_size': int(args.batch_size)
            }
                    
            save_path = args.model_save_path.replace('/model', '') + '/' + args.dataset
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_path = save_path + '/' + args.method
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_path = save_path + '/' + str(args.state_keep_prob)
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            model_save_path = save_path + '/model'

            pure_LSTM = AWLSTM(
                data_path=args.path,
                model_path=args.model_path,
                model_save_path=model_save_path,
                parameters=parameters,
                steps=args.step,
                epochs=args.epoch, batch_size=args.batch_size, gpu=args.gpu,
                tra_date=args.tra_date, val_date=args.val_date, tes_date=args.tes_date, att=args.att,
                hinge=args.hinge_lose, fix_init=args.fix_init, adv=args.adv,
                reload=args.reload,
                dropout_activation_function = 'avg'
            )

            runs = 5
            runs_arr = [*range(runs + 1)][1:]

            perf_valid_arr = []
            perf_test_arr = []
            
            for r in runs_arr:
                run_save_path = save_path + '/' + str(r)

                if not os.path.exists(run_save_path):
                    os.mkdir(run_save_path)  

                model_save_path = run_save_path + '/model'
                
                pure_LSTM.model_save_path = model_save_path
                best_valid_perf, best_test_perf, best_iterations = pure_LSTM.train_monte_carlo_dropout(return_perf=True, return_pred=False, iterations_arr=args.dropout_iterations_arr)

                perf_valid_arr.append(best_valid_perf)
                perf_test_arr.append(best_test_perf)  
                val_pre_std = list(filter(lambda x: x['measure'] == 'std', best_valid_perf['prob_arr']))[0]['val']  
                val_pre_var =  list(filter(lambda x: x['measure'] == 'var', best_valid_perf['prob_arr']))[0]['val']  
                val_pre_entr = list(filter(lambda x: x['measure'] == 'entr', best_valid_perf['prob_arr']))[0]['val']  
                tes_pre_std = list(filter(lambda x: x['measure'] == 'std', best_test_perf['prob_arr']))[0]['val']  
                tes_pre_var = list(filter(lambda x: x['measure'] == 'var', best_test_perf['prob_arr']))[0]['val']  
                tes_pre_entr = list(filter(lambda x: x['measure'] == 'entr', best_test_perf['prob_arr']))[0]['val']  
                
                val_avg_std = np.average(val_pre_std)
                val_avg_var = np.average(val_pre_var)
                val_avg_entr = np.average(val_pre_entr)
                tes_avg_std = np.average(tes_pre_std)
                tes_avg_var = np.average(tes_pre_var)
                tes_avg_entr = np.average(tes_pre_entr)

                perf_dict = {
                        'method': [args.method],
                        'dataset': [args.dataset],
                        'test acc': [best_test_perf['acc'] * 100],
                        'test mcc': [best_test_perf['mcc']],
                        'test ll': [best_test_perf['ll']],
                        'test rs': [best_test_perf['rs']],
                        'test ps' : [best_test_perf['ps']],
                        'test avg std': [tes_avg_std],
                        'test avg var': [tes_avg_var],
                        'test avg entr': [tes_avg_entr],
                        'valid acc': [best_valid_perf['acc'] * 100],
                        'valid mcc': [best_valid_perf['mcc']],
                        'valid ll': [best_valid_perf['ll']],
                        'valid rs': [best_valid_perf['rs']],
                        'valid ps' : [best_valid_perf['ps']],
                        'valid avg std': [val_avg_std],
                        'valid avg var': [val_avg_var],
                        'valid avg entr': [val_avg_entr],
                        'dropout' : [args.state_keep_prob],
                        'run': [r],
                        'iterations': [best_iterations],
                        'run save path': [run_save_path]
                    }

                df = pd.DataFrame(perf_dict)
                if perf_df is None:
                    perf_df = df
                else:
                    perf_df = pd.concat([perf_df, df])            

                if not os.path.exists('experiment1'):
                    os.mkdir('experiment1')
                if not os.path.exists('experiment1/dropout'):
                    os.mkdir('experiment1/dropout')
                perf_df.to_csv('experiment1/dropout/perf_dropout_results.csv', index = False)
                
                with open(run_save_path + '/perf_df.pkl', 'wb') as perf_df_file:
                    pickle.dump(perf_df, perf_df_file)
                #dfi.export(perf_df,"experiment1/perf_dropout_results.png")

                with open(run_save_path + '/best_valid_perf.pkl', 'wb') as best_valid_perf_file:
                    pickle.dump(best_valid_perf, best_valid_perf_file)
                with open(run_save_path + '/best_test_perf.pkl', 'wb') as best_test_perf_file:
                    pickle.dump(best_test_perf, best_test_perf_file)
                
            valid_acc_list = list(map(lambda x: x['acc'], perf_valid_arr))
            valid_mcc_list = list(map(lambda x: x['mcc'], perf_valid_arr))
            valid_ll_list = list(map(lambda x: x['ll'], perf_valid_arr))
            valid_rs_list = list(map(lambda x: x['rs'], perf_valid_arr))
            valid_ps_list = list(map(lambda x: x['ps'], perf_valid_arr))

            valid_prob_arr_select_many = [item for row in list(map(lambda x: x['prob_arr'], perf_valid_arr)) for item in row]
            valid_pre_std_list = list(map(lambda x: x['val'], list(filter(lambda x: x['measure'] == 'std', valid_prob_arr_select_many))))
            valid_pre_var_list = list(map(lambda x: x['val'], list(filter(lambda x: x['measure'] == 'var', valid_prob_arr_select_many))))
            valid_pre_ent_list = list(map(lambda x: x['val'], list(filter(lambda x: x['measure'] == 'entr', valid_prob_arr_select_many))))

            avg_valid_acc = np.average(np.array(valid_acc_list)) * 100
            avg_valid_mcc = np.average(np.array(valid_mcc_list))
            avg_valid_ll = np.average(np.array(valid_ll_list))
            std_valid_ll = np.std(np.array(valid_ll_list), ddof=1) / np.sqrt(np.size(np.array(valid_ll_list)))
            avg_valid_rs = np.average(np.array(valid_rs_list))
            avg_valid_ps = np.average(np.array(valid_ps_list))
            avg_valid_std = np.average(np.array(valid_pre_std_list))
            avg_valid_var = np.average(np.array(valid_pre_var_list))
            avg_valid_ent = np.average(np.array(valid_pre_ent_list))

            test_acc_list = list(map(lambda x: x['acc'], perf_test_arr))
            test_mcc_list = list(map(lambda x: x['mcc'], perf_test_arr))
            test_ll_list = list(map(lambda x: x['ll'], perf_test_arr))
            test_rs_list = list(map(lambda x: x['rs'], perf_test_arr))
            test_ps_list = list(map(lambda x: x['ps'], perf_test_arr))
            test_prob_arr_select_many = [item for row in list(map(lambda x: x['prob_arr'], perf_test_arr)) for item in row]
            test_pre_std_list = list(map(lambda x: x['val'], list(filter(lambda x: x['measure'] == 'std', test_prob_arr_select_many))))
            test_pre_var_list = list(map(lambda x: x['val'], list(filter(lambda x: x['measure'] == 'var', test_prob_arr_select_many))))
            test_pre_ent_list = list(map(lambda x: x['val'], list(filter(lambda x: x['measure'] == 'entr', test_prob_arr_select_many))))


            avg_test_acc = np.average(np.array(test_acc_list)) * 100
            avg_test_mcc = np.average(np.array(test_mcc_list))
            avg_test_ll = np.average(np.array(test_ll_list))
            std_test_ll = np.std(np.array(test_ll_list), ddof=1) / np.sqrt(np.size(np.array(test_ll_list)))
            avg_test_rs = np.average(np.array(test_rs_list))
            avg_test_ps = np.average(np.array(test_ps_list))
            avg_test_std = np.average(np.array(test_pre_std_list))
            avg_test_var = np.average(np.array(test_pre_var_list))
            avg_test_ent = np.average(np.array(test_pre_ent_list))

            avg_acc = np.average(np.array([avg_test_acc, avg_valid_acc]))
            avg_mcc = np.average(np.array([avg_test_mcc, avg_valid_mcc]))
            avg_std_err_ll = np.average(np.array([std_test_ll, std_valid_ll]))
            avg_rs = np.average(np.array([avg_test_rs, avg_valid_rs]))
            avg_ps = np.average(np.array([avg_test_ps, avg_valid_ps]))
            avg_std = np.average(np.array([avg_test_std, avg_valid_std]))
            avg_var = np.average(np.array([avg_test_var, avg_valid_var]))
            avg_ent = np.average(np.array([avg_test_ent, avg_valid_ent]))

            perf_dict_2 = {
                        'method': [args.method],
                        'dataset': [args.dataset],
                        'dropout' : [args.state_keep_prob],
                        'avg test acc': [avg_test_acc],
                        'avg test mcc': [avg_test_mcc],
                        'avg test ll': [avg_test_ll],
                        'std error test ll': [std_test_ll],
                        'avg test rs' : [avg_test_rs],
                        'avg test ps' : [avg_test_ps],
                        'avg test std' : [avg_test_std],
                        'avg test var' : [avg_test_var],
                        'avg test ent' : [avg_test_ent],
                        'avg valid acc': [avg_valid_acc],
                        'avg valid mcc': [avg_valid_mcc],
                        'avg valid ll': [avg_valid_ll],
                        'std error valid ll': [std_valid_ll],
                        'avg valid rs' : [avg_valid_rs],
                        'avg valid ps' : [avg_valid_ps],
                        'avg valid std' : [avg_valid_std],
                        'avg valid var' : [avg_valid_var],
                        'avg valid ent' : [avg_valid_ent],
                        'avg acc': [avg_acc],
                        'avg mcc': [avg_mcc],
                        'avg std error ll': [avg_std_err_ll],
                        'avg rs' : [avg_rs],
                        'avg ps' : [avg_ps],
                        'avg std' : [avg_std],
                        'avg var' : [avg_var],                          
                        'avg ent' : [avg_ent]
                    }

            df_2 = pd.DataFrame(perf_dict_2)

            if perf_df2 is None:
                perf_df2 = df_2
            else:
                perf_df2 = pd.concat([perf_df2, df_2])         

            if not os.path.exists('experiment1'):
                os.mkdir('experiment1')
            if not os.path.exists('experiment1/dropout'):
                os.mkdir('experiment1/dropout')
            perf_df2.to_csv('experiment1/dropout/perf_dropout_grouped_results.csv', index = False)
            #dfi.export(perf_df2,"experiment1/perf_dropout_grouped_results.png")

            with open(save_path  + '/perf_df2.pkl', 'wb') as perf_df2_file:
                pickle.dump(perf_df2, perf_df2_file)

    return perf_df, perf_df2

def run_experiment_1_dropout_uncertainty(threshold_strategy_arr):
    benchmark_df = pd.read_csv('./experiment1/dropout/perf_dropout_results.csv')
    prob_arr = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
    perf_df = None

    # model = benchmark_df[(benchmark_df['dataset'] == 'kdd17') & (benchmark_df['method'] == 'Adv-ALSTM') & (benchmark_df['dropout'] == 0.05) & (benchmark_df['run'] == 5)]
    # model = model.reset_index()
    # run_save_path = model['run save path'][0]
    # with open(run_save_path + '/perf_test_arr.pkl', "rb") as input_file:
    #     perf_test_arr = pickle.load(input_file)
    # with open(run_save_path + '/perf_valid_arr.pkl', "rb") as input_file:
    #     perf_valid_arr = pickle.load(input_file)

    # for i, x in enumerate(perf_test_arr):
    #     with open(run_save_path.replace('/5', '/' + str(i + 1)) + '/best_test_perf.pkl', 'wb') as perf_df_file:
    #         pickle.dump(perf_test_arr[i], perf_df_file)
    # for i, x in enumerate(perf_valid_arr):
    #     with open(run_save_path.replace('/5', '/' + str(i + 1)) + '/best_valid_perf.pkl', 'wb') as perf_df_file:
    #         pickle.dump(perf_valid_arr[i], perf_df_file)

    for index, row in benchmark_df.iterrows():
        run_save_path = row['run save path']
        with open(run_save_path + '/best_test_perf.pkl', "rb") as input_file:
            best_test_perf = pickle.load(input_file)
        with open(run_save_path + '/best_valid_perf.pkl', "rb") as input_file:
            best_valid_perf = pickle.load(input_file)
        for mi, m in enumerate(best_valid_perf['prob_arr']):
            
            val_pre_prob = best_valid_perf['prob_arr'][mi]['val']
            tes_pre_prob = best_test_perf['prob_arr'][mi]['val']
            prob_measure = best_valid_perf['prob_arr'][mi]['measure']

            #for i, a in enumerate(best_valid_perf['prob_arr']):
            for st in threshold_strategy_arr:
                for p in prob_arr:
                    valid_prob_arr = np.copy(val_pre_prob)
                    max_valid_prob = np.max(valid_prob_arr)

                    valid_pred = np.copy(best_valid_perf['pred'])
                    valid_gt = np.copy(best_valid_perf['gt'])

                    valid_pred_mask_array = []
                    valid_gt_mask_array = []

                    valid_filtered = 0
                    valid_filtered_successfully = 0
                    valid_success_score = 0
                    valid_failure_score = 0
                    for i, x in np.ndenumerate(valid_prob_arr):
                        filt = True
                        #perc = (x/max_valid_prob) * 100
                        #max_valid_perc = (100-(p*100))
                        #if perc > max_valid_perc and valid_pred[i] == 0:
                        if x > (1-p) and (st == 'normal' or (st == 'true' and valid_pred[i] == 1) or (st == 'false' and valid_pred[i] == 0)):
                            filt = False
                            valid_filtered = valid_filtered + 1
                            if valid_gt[i] != valid_pred[i]:
                                valid_filtered_successfully = valid_filtered_successfully + 1
                        if ((valid_gt[i] == valid_pred[i] and filt == True) or (valid_gt[i] != valid_pred[i] and filt == False)):
                            valid_success_score = valid_success_score + 1 
                        else:
                            valid_failure_score = valid_failure_score + 1
                        pred = valid_pred[i]
                        gt = valid_gt[i]
                        valid_pred_mask_array.append(filt)
                        valid_gt_mask_array.append(filt)

                    valid_pred_filter = valid_pred[valid_pred_mask_array]
                    valid_gt_filter = valid_gt[valid_gt_mask_array]

                    total_valid_predictions = valid_gt_filter.shape[0]
                    if valid_pred_filter.shape[0] == 0:
                        valid_pred_filter = np.zeros((valid_pred.shape[0], valid_pred.shape[1]))
                    if valid_gt_filter.shape[0] == 0:
                        valid_gt_filter = np.zeros((valid_gt.shape[0], valid_gt.shape[1]))

                    cur_valid_perf = evaluate(valid_pred_filter, valid_gt_filter, best_valid_perf['hinge'], additional_metrics=best_valid_perf['additional_metrics'])
                    
                    uncertainty_run_save_path_uncertainty = run_save_path + '/' + str(p)
                    if not os.path.exists(uncertainty_run_save_path_uncertainty):
                        os.mkdir(uncertainty_run_save_path_uncertainty)  

                    with open(uncertainty_run_save_path_uncertainty + '/valid_pred_filter.pkl', 'wb') as valid_pred_filter_file:
                        pickle.dump(valid_pred_filter, valid_pred_filter_file)
                    with open(uncertainty_run_save_path_uncertainty + '/valid_gt_filter.pkl', 'wb') as valid_gt_filter_file:
                        pickle.dump(valid_gt_filter, valid_gt_filter_file)

                    test_prob_arr = np.copy(tes_pre_prob)
                    #max_test_prob = np.max(test_prob_arr)
                    test_pred = np.copy(best_test_perf['pred'])
                    test_gt = np.copy(best_test_perf['gt'])

                    test_pred_mask_array = []
                    test_gt_mask_array = []

                    test_filtered = 0
                    test_filtered_successfully = 0
                    test_success_score = 0
                    test_failure_score = 0
                    for i, x in np.ndenumerate(test_prob_arr):
                        filt = True
                        #perc = (x/max_test_prob) * 100
                        #max_test_perc = (100-(p*100))
                        #if perc > max_test_perc and test_pred[i] == 0 :
                        if x > (1-p) and (st == 'normal' or (st == 'true' and test_pred[i] == 1) or (st == 'false' and test_pred[i] == 0)):
                            filt = False
                            test_filtered = test_filtered + 1
                            if test_gt[i] != test_pred[i]:
                                test_filtered_successfully = test_filtered_successfully + 1
                        if ((test_gt[i] == test_pred[i] and filt == True) or (test_gt[i] != test_pred[i] and filt == False)):
                            test_success_score = test_success_score + 1 
                        else:
                            test_failure_score = test_failure_score + 1
                        test_pred_mask_array.append(filt)
                        test_gt_mask_array.append(filt)

                    test_pred_filter = test_pred[test_pred_mask_array]
                    test_gt_filter = test_gt[test_gt_mask_array]
                    total_test_predictions = test_pred_filter.shape[0]
                    if test_pred_filter.shape[0] == 0:
                        test_pred_filter = np.zeros((test_pred.shape[0], test_pred.shape[1]))
                    if test_gt_filter.shape[0] == 0:
                        test_gt_filter = np.zeros((test_gt.shape[0], test_gt.shape[1]))

                    cur_test_perf = evaluate(test_pred_filter, test_gt_filter, best_test_perf['hinge'], additional_metrics=best_test_perf['additional_metrics'])
                    with open(run_save_path + '/' + str(p) + '/test_pred_filter.pkl', 'wb') as test_pred_filter_file:
                        pickle.dump(test_pred_filter, test_pred_filter_file)
                    with open(run_save_path + '/' + str(p) + '/test_gt_filter.pkl', 'wb') as test_gt_filter_file:
                        pickle.dump(test_gt_filter, test_gt_filter_file)

                    val_avg_prob = np.average(val_pre_prob)
                    tes_avg_prob = np.average(tes_pre_prob)

                    perf_dict = {
                            'method': [row['method']],
                            'dataset': [row['dataset']],
                            'test acc': [cur_test_perf['acc'] * 100],
                            'test mcc': [cur_test_perf['mcc']],
                            'test ll': [cur_test_perf['ll']],
                            'test rs': [cur_test_perf['rs']],
                            'test ps' : [cur_test_perf['ps']],
                            'test avg prob': [tes_avg_prob],
                            'valid acc': [cur_valid_perf['acc'] * 100],
                            'valid mcc': [cur_valid_perf['mcc']],
                            'valid ll': [cur_valid_perf['ll']],
                            'valid rs': [cur_valid_perf['rs']],
                            'valid ps' : [cur_valid_perf['ps']],
                            'valid avg prob': [val_avg_prob],
                            'prob measure': [prob_measure],
                            'dropout' : [row['dropout']],
                            'run': [row['run']],
                            'prob confidence threshold': [p],
                            'thresholding strategy': [st],
                            'iterations': [row['iterations']],
                            'run save path': [row['run save path']],
                            'total test predictions': [total_test_predictions],
                            'filtered test predictions': [test_filtered],
                            'filtered test predictions sucessfully': [test_filtered_successfully],
                            'filtered test predictions sucessful ratio': [test_filtered and test_filtered_successfully / test_filtered or 0],
                            'test success score': [test_success_score],
                            'test failure score': [test_failure_score],
                            'test score':[test_success_score / (test_success_score + test_failure_score)],
                            'total valid predictions': [total_valid_predictions],
                            'filtered valid predictions': [valid_filtered],
                            'filtered valid predictions sucessfully': [valid_filtered_successfully],
                            'filtered valid predictions sucessful ratio': [valid_filtered and valid_filtered_successfully / valid_filtered or 0],
                            'valid success score': [valid_success_score],
                            'valid failure score': [valid_failure_score],
                            'valid score': [valid_success_score / (valid_success_score + valid_failure_score)]
                        }

                    df = pd.DataFrame(perf_dict)
                    if perf_df is None:
                        perf_df = df
                    else:
                        perf_df = pd.concat([perf_df, df])            

                    if not os.path.exists('experiment1'):
                        os.mkdir('experiment1')
                    if not os.path.exists('experiment1/dropout_uncertainty'):
                        os.mkdir('experiment1/dropout_uncertainty')

                    perf_df.to_csv('experiment1/dropout_uncertainty/perf_dropout_results.csv', index = False)