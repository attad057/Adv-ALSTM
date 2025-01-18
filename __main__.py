from pred_lstm_replication import run_replication, t_test_summary_array
from pred_lstm_experiment_1 import run_experiment_1_dropout, run_experiment_1_dropout_uncertainty             
from pred_lstm_experiment_2 import run_experiment_2_replication, run_experiment_2_dropout
from report import report_dropout_best_runs, report_best_dropout_model_comparison_test, report_dropout_grouped_runs, report_best_dropout_run_binary_graphs, report_compare_ticker_replication, report_uncertainty_heatmap, report_dropout_uncertainty, report_pr_curves, report_best_uncertainty_model_comparison, report_best_model_comparison_test
import argparse

if __name__ == '__main__':
    desc = 'the lstm model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-p', '--path', help='path of pv data', type=str,
                        default='./data/stocknet-dataset/price/ourpped')
    parser.add_argument('-l', '--seq', help='length of history', type=int,
                        default=5)
    parser.add_argument('-u', '--unit', help='number of hidden units in lstm',
                        type=int, default=32)
    parser.add_argument('-l2', '--alpha_l2', type=float, default=1e-2,
                        help='alpha for l2 regularizer')
    parser.add_argument('-la', '--beta_adv', type=float, default=1e-2,
                        help='beta for adverarial loss')
    parser.add_argument('-le', '--epsilon_adv', type=float, default=1e-2,
                        help='epsilon to control the scale of noise')
    parser.add_argument('-s', '--step', help='steps to make prediction',
                        type=int, default=1)
    parser.add_argument('-b', '--batch_size', help='batch size', type=int,
                        default=1024)
    parser.add_argument('-e', '--epoch', help='epoch', type=int, default=150)
    parser.add_argument('-r', '--learning_rate', help='learning rate',
                        type=float, default=1e-2)
    parser.add_argument('-g', '--gpu', type=int, default=0, help='use gpu')
    parser.add_argument('-q', '--model_path', help='path to load model',
                        type=str, default='./saved_model/acl18_alstm/exp')
    parser.add_argument('-qs', '--model_save_path', type=str, help='path to save model',
                        default='./tmp/model')
    parser.add_argument('-o', '--action', type=str, default='train',
                        help='train, test, pred, replicate')
    parser.add_argument('-m', '--model', type=str, default='pure_lstm',
                        help='pure_lstm, di_lstm, att_lstm, week_lstm, aw_lstm')
    parser.add_argument('-f', '--fix_init', type=int, default=0,
                        help='use fixed initialization')
    parser.add_argument('-a', '--att', type=int, default=1,
                        help='use attention model')
    parser.add_argument('-w', '--week', type=int, default=0,
                        help='use week day data')
    parser.add_argument('-v', '--adv', type=int, default=0,
                        help='adversarial training')
    parser.add_argument('-hi', '--hinge_lose', type=int, default=1,
                        help='use hinge lose')
    parser.add_argument('-rl', '--reload', type=int, default=0,
                        help='use pre-trained parameters')
    parser.add_argument('-sp', '--state_keep_prob', type=float, default=1,
                    help='Apply dropout to each layer')
    args = parser.parse_args()
    dataset = 'stocknet' if 'stocknet' in args.path else 'kdd17' if 'kdd17' in args.path else ''
    method = ''
    args.action = 'experiment2_dropout'
    
    if args.action == 'replication':
        predefined_args = [ 
        {
            #-p
            'path': './data/stocknet-dataset/price/ourpped',
            #-a
            'att': 0,
            #-l
            'seq': 10,
            #-u
            'unit': 32,
            #-l2
            'alpha_l2': 10,
            #-f
            'fix_init': 1,
            #-v
            'adv': 0,
            #-rl
            'reload': 0,
            #-la
            'beta_adv': 1e-2,
            #-le
            'epsilon_adv': 1e-2,
            #-q
            'model_path': './saved_model/acl18_alstm/exp',
            'model_save_path': './tmp/model',
            'method': 'LSTM',
            'dataset': 'stocknet'
        },
        {
            #-p
            'path': './data/stocknet-dataset/price/ourpped',
            #-a
            'att': 1,
            #-l
            'seq': 5,
            #-u
            'unit': 4,
            #-l2
            'alpha_l2': 1,
            #-f
            'fix_init': 1,
            #-v
            'adv': 0,
            #-rl
            'reload': 0,
            #-la
            'beta_adv': 1e-2,
            #-le
            'epsilon_adv': 1e-2,
            #-q
            'model_path': './saved_model/acl18_alstm/exp',
            'model_save_path': './tmp/model',
            'method': 'ALSTM',
            'dataset': 'stocknet'
        },
        {
            #-p
            'path': './data/stocknet-dataset/price/ourpped',
            #-a
            'att': 1,
            #-l
            'seq': 5,
            #-u
            'unit': 4,
            #-l2
            'alpha_l2': 1,
            #-f
            'fix_init': 0,
            #-v
            'adv': 1,
            #-rl
            'reload': 1,
            #-la
            'beta_adv': 0.01,
            #-le
            'epsilon_adv': 0.05,
            'model_path': './saved_model/acl18_alstm/exp',
            'model_save_path': './tmp/model',
            'method': 'Adv-ALSTM',
            'dataset': 'stocknet'
        },   
        {
            #-p
            'path': './data/kdd17/ourpped/',
            #-a
            'att': 0,
            #-l
            'seq': 5,
            #-u
            'unit': 4,
            #-l2
            'alpha_l2': 0.001,
            #-f
            'fix_init': 1,
            #-v
            'adv': 0,
            #-rl
            'reload': 0,
            #-la
            'beta_adv': 1e-2,
            #-le
            'epsilon_adv': 1e-2,
            #-q
            'model_path': './saved_model/kdd17_alstm/model',
            'model_save_path': './tmp/model',
            'method': 'LSTM',
            'dataset': 'kdd17'
        },  
        {
            #-p
            'path': './data/kdd17/ourpped/',
            #-a
            'att': 1,
            #-l
            'seq': 15,
            #-u
            'unit': 16,
            #-l2
            'alpha_l2': 0.001,
            #-f
            'fix_init': 1,
            #-v
            'adv': 0,
            #-rl
            'reload': 0,
            #-la
            'beta_adv': 1e-2,
            #-le
            'epsilon_adv': 1e-2,
            #-q
            'model_path': './saved_model/kdd17_alstm/model',
            'model_save_path': './tmp/model',
            'method': 'ALSTM',
            'dataset': 'kdd17'
        },
        {
            #-p
            'path': './data/kdd17/ourpped/',
            #-a
            'att': 1,
            #-l
            'seq': 15,
            #-u
            'unit': 16,
            #-l2
            'alpha_l2': 0.001,
            #-f
            'fix_init': 1,
            #-v
            'adv': 1,
            #-rl
            'reload': 1,
            #-la
            'beta_adv': 0.05,
            #-le
            'epsilon_adv':  0.001,
            'model_path': './saved_model/kdd17_alstm/model',
            'model_save_path': './tmp/model',
            'method': 'Adv-ALSTM',
            'dataset': 'kdd17'
        }]

        run_replication(predefined_args, args)
    elif args.action == 'experiment1_dropout':
        predefined_args = [
        {
            #-p
            'path': './data/stocknet-dataset/price/ourpped',
            #-a
            'att': 1,
            #-l
            'seq': 5,
            #-u
            'unit': 4,
            #-l2
            'alpha_l2': 1,
            #-f
            'fix_init': 0,
            #-v
            'adv': 1,
            #-rl
            'reload': 1,
            #-la
            'beta_adv': 0.01,
            #-le
            'epsilon_adv': 0.05,
            'model_path': './saved_model/acl18_alstm/exp',
            'model_save_path': './experiment1/dropout/tmp/model',
            'method': 'Adv-ALSTM',
            'dataset': 'stocknet',
            'batch_size': 1024
        },
        {
            #-p
            'path': './data/kdd17/ourpped/',
            #-a
            'att': 1,
            #-l
            'seq': 15,
            #-u
            'unit': 16,
            #-l2
            'alpha_l2': 0.001,
            #-f
            'fix_init': 1,
            #-v
            'adv': 1,
            #-rl
            'reload': 1,
            #-la
            'beta_adv': 0.05,
            #-le
            'epsilon_adv':  0.001,
            'model_path': './saved_model/kdd17_alstm/model',
            'model_save_path': './experiment1/dropout/tmp/model',
            'method': 'Adv-ALSTM',
            'dataset': 'kdd17',
            'batch_size': 1024
        }]

        perf_df = None
        perf_df2 = None

        args.state_keep_prob_arr = [0.05, 0.1, 0.25, 0.5, 0.95, 0.9, 0.75]
        args.dropout_iterations_arr = [500, 1000, 2000]

        perf_df, perf_df2 = run_experiment_1_dropout(predefined_args, args)
    elif args.action == 'experiment1_dropout_uncertainty':
        run_experiment_1_dropout_uncertainty(['normal', 'true', 'false'])
    elif args.action == 'experiment2_dropout':
        run_experiment_2_dropout()
    elif args.action == 'experiment2_replication':
        predefined_args = [  
                {
            #-p
            'path': './data/stocknet-dataset/price/ourpped',
            #-a
            'att': 1,
            #-l
            'seq': 5,
            #-u
            'unit': 4,
            #-l2
            'alpha_l2': 1,
            #-f
            'fix_init': 0,
            #-v
            'adv': 1,
            #-rl
            'reload': 1,
            #-la
            'beta_adv': 0.01,
            #-le
            'epsilon_adv': 0.05,
            'model_path': './saved_model/acl18_alstm/exp',
            'model_save_path': './experiment1/dropout/tmp/model',
            'method': 'Adv-ALSTM',
            'dataset': 'stocknet',
            'batch_size': 1024
        },
        {
            #-p
            'path': './data/kdd17/ourpped/',
            #-a
            'att': 1,
            #-l
            'seq': 15,
            #-u
            'unit': 16,
            #-l2
            'alpha_l2': 0.001,
            #-f
            'fix_init': 1,
            #-v
            'adv': 1,
            #-rl
            'reload': 1,
            #-la
            'beta_adv': 0.05,
            #-le
            'epsilon_adv':  0.001,
            'model_path': './saved_model/kdd17_alstm/model',
            'model_save_path': './experiment1/dropout/tmp/model',
            'method': 'Adv-ALSTM',
            'dataset': 'kdd17',
            'batch_size': 1024
        }
       ]

        run_experiment_2_replication(predefined_args, args)
    elif args.action == 'report':
        #add f1 score to test table

        #T-test summary for replication ALSTM benchmark vs replication
        # mean1_array, std1_array, n1_array = [0.1043, 0.0261, 54.90, 51.94], [1e-2, 1e-2, 7e-1, 7e-1], [5, 5, 5, 5]  # Group 1: mean, standard deviation, sample size
        # mean2_array, std2_array, n2_array = [0.0910, 0.0150, 54.44, 51.43], [2e-2, 7e-3, 9e-1, 4e-1], [5, 5, 5, 5]  # Group 2: mean, standard deviation, sample size
        # t_test_summary_array(mean1_array, std1_array, n1_array, mean2_array, std2_array, n2_array)

        #Experiment 1 dropout
        #1. Line graph describing aggregate statistics for dropout experiment 1
        # report_dropout_grouped_runs('valid acc', 'Valid ACC Score', 'ACC Score', True)
        # report_dropout_grouped_runs('valid mcc', 'Valid MCC Score', 'MCC Score', False)
        
        #2. Line graph of best runs for dropout experiment 1
        #report_dropout_best_runs('valid acc', 'Valid ACC %', 'ACC %')
        #report_dropout_best_runs('valid mcc', 'Valid MCC Score', 'MCC Score')

        #Experiment 1 dropout uncertainty
        #1. Bar graph for best uncertainty model comparison
        #report_best_uncertainty_model_comparison('valid acc', 'Valid ACC %', True)
        #report_best_uncertainty_model_comparison('valid mcc', 'Valid MCC %', False)

        #2. Heatmap for uncertainty measures
        #report_uncertainty_heatmap('valid acc', 'Valid ACC %')

        #3. Line graph of acc vs uncertainty thresholds
        #report_dropout_uncertainty('valid acc', 'Valid ACC %', 'ACC %', ['#87CEEB', '#4169E1', '#000080'])
        #report_dropout_uncertainty('valid mcc', 'Valid MCC Score', 'MCC Score', ['#228B22', '#98FB98', '#808000'])
        
        #4. Bar Graph for best model comparison
        #report_best_model_comparison_test('test acc', 'Test ACC %', True)
        #report_best_model_comparison_test('test mcc', 'Test MCC %', False)
        report_best_dropout_model_comparison_test('test acc', 'Test ACC %', True)
        report_best_dropout_model_comparison_test('test mcc', 'Test MCC Score', False)


        #report_pr_curves('valid acc', 'Valid ACC %', 'ACC %', ['#87CEEB', '#4169E1', '#000080'])
        #report_best_dropout_run_binary_graphs('valid acc')
        #report_compare_ticker_replication('valid acc', 'Valid ACC %')

        #density plot ? (not needed)
        #correlation heatmap
        #correlation graph mcc / acc
        #statistical test ?

        #correlation heatmap return / sharpes ratio
        #boxplot of stocks
        #uncertainty heatmap
        #accumultive returns
        #accumlative Sharpes ratio
    else:
        print(args)

        dropout_activation_function = None
        if (args.state_keep_prob < 1):
            dropout_activation_function = 'avg'

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
                'act': args.action,
                'state_keep_prob': float(args.state_keep_prob),
                'batch_size': int(args.batch_size)
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
            dropout_activation_function = dropout_activation_function
        )
        if args.action == 'train':
                pure_LSTM.train()      
        elif args.action == 'test':
            pure_LSTM.test()
        elif args.action == 'pred':
            pure_LSTM.predict_record()
        elif args.action == 'adv':
            pure_LSTM.predict_adv()
        elif args.action == 'latent':
            pure_LSTM.get_latent_rep()