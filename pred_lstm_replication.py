from pred_lstm import AWLSTM               
import pandas as pd
import os
import numpy as np
import dataframe_image as dfi
import pickle     

def run_replication(predefined_args, args):    
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
            if args.dataset == 'stocknet':
                tra_date = '2014-01-02'
                val_date = '2015-08-03'
                tes_date = '2015-10-01'
            elif args.dataset == 'kdd17':
                tra_date = '2007-01-03'
                val_date = '2015-01-02'
                tes_date = '2016-01-04'
            else:
                print('unexpected dataset: %s' % dataset)
                exit(0)
            print(args)
            parameters = {
                'seq': int(args.seq),
                'unit': int(args.unit),
                'alp': float(args.alpha_l2),
                'bet': float(args.beta_adv),
                'eps': float(args.epsilon_adv),
                'lr': float(args.learning_rate)
            }

            save_path = args.model_save_path.replace('/model', '') + '/' + args.dataset
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_path = save_path + '/' + args.method
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            args.model_save_path = save_path + '/model'
                    
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
                dropout_activation_function = None
            )

            runs = 5
            runs_arr = [*range(runs + 1)][1:]    
            pred_valid_arr = []
            pred_test_arr = []

            for r in runs_arr:
                run_save_path = save_path + '/' + str(r)
                if not os.path.exists(run_save_path):
                    os.mkdir(run_save_path)  
                args.model_save_path = run_save_path + '/model'
                pure_LSTM.model_save_path = args.model_save_path

                best_valid_perf, best_test_perf = pure_LSTM.train(return_perf=True, return_pred=False)
                pred_valid_arr.append(best_valid_perf)
                pred_test_arr.append(best_test_perf)
                perf_dict = {
                        'method': [args.method],
                        'dataset': [args.dataset],
                        'test acc': [best_test_perf['acc'] * 100],
                        'test mcc': [best_test_perf['mcc']],
                        'valid acc': [best_valid_perf['acc'] * 100],
                        'valid mcc': [best_valid_perf['mcc']],
                        'run': [r]
                    }

                df = pd.DataFrame(perf_dict)
                if perf_df is None:
                    perf_df = df
                else:
                    perf_df = pd.concat([perf_df, df])            

                if not os.path.exists('replication'):
                    os.mkdir('replication')
                perf_df.to_csv('replication/perf_results.csv', index = False)
                #dfi.export(perf_df,"replication/perf_results.png")
                with open('replication/perf_results.pkl', 'wb') as perf_df_file:
                    pickle.dump(perf_df, perf_df_file)

            valid_acc_list = list(map(lambda x: x['acc'], pred_valid_arr))
            valid_mcc_list = list(map(lambda x: x['mcc'], pred_valid_arr))
            valid_ll_list = list(map(lambda x: x['ll'], pred_valid_arr))

            avg_valid_acc = np.average(np.array(valid_acc_list)) * 100
            avg_valid_mcc = np.average(np.array(valid_mcc_list))
            avg_valid_ll = np.average(np.array(valid_ll_list))
            std_valid_ll = np.std(np.array(valid_ll_list), ddof=1) / np.sqrt(np.size(np.array(valid_ll_list)))

            test_acc_list = list(map(lambda x: x['acc'], pred_test_arr))
            test_mcc_list = list(map(lambda x: x['mcc'], pred_test_arr))
            test_ll_list = list(map(lambda x: x['ll'], pred_test_arr))

            avg_test_acc = np.average(np.array(test_acc_list)) * 100
            avg_test_mcc = np.average(np.array(test_mcc_list))
            avg_test_ll = np.average(np.array(test_ll_list))
            std_test_ll= np.std(np.array(test_ll_list), ddof=1) / np.sqrt(np.size(np.array(test_ll_list)))
            avg_acc = np.average(np.array([avg_test_acc, avg_valid_acc]))
            avg_mcc = np.average(np.array([avg_test_mcc, avg_valid_mcc]))
            avg_std_err_ll = np.average(np.array([std_test_ll, std_valid_ll]))

            perf_dict_2 = {
                        'method': [args.method],
                        'dataset': [args.dataset],
                        'avg test acc': [avg_test_acc],
                        'avg test mcc': [avg_test_mcc],
                        'avg test ll': [avg_test_ll],
                        'std error test ll': [std_test_ll],
                        'avg valid acc': [avg_valid_acc],
                        'avg valid mcc': [avg_valid_mcc],
                        'avg valid ll': [avg_valid_ll],
                        'std error valid ll': [std_valid_ll],
                        'avg acc': [avg_acc],
                        'avg mcc': [avg_mcc],
                        'avg std error ll': [avg_std_err_ll]
                    }
            df_2 = pd.DataFrame(perf_dict_2)

            if perf_df2 is None:
                perf_df2 = df_2
            else:
                perf_df2 = pd.concat([perf_df2, df_2])         

            if not os.path.exists('replication'):
                os.mkdir('replication')
            perf_df2.to_csv('replication/perf_grouped_results.csv', index = False)
            #dfi.export(perf_df2,"replication/perf_grouped_results.png")
                      
            with open('replication/perf_grouped_results.pkl', 'wb') as perf_df2_file:
                pickle.dump(perf_df2, perf_df2_file)