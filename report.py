import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
import pickle  
import math

def adjust_shade(hex_color, factor=0.8):
    # Convert HEX to RGB
    rgb = [int(hex_color[i:i+2], 16) for i in (1, 3, 5)]  # Extract RGB components
    # Darken or lighten each RGB component
    adjusted_rgb = [max(min(int(c * factor), 255), 0) for c in rgb]
    # Convert adjusted RGB back to HEX
    return '#' + ''.join([f'{c:02x}' for c in adjusted_rgb])
    
def report_best_dropout_run_binary_graphs(metric_column):
    dropout_df = pd.read_csv('./experiment1/dropout/perf_dropout_results.csv')
    distinct_datasets = dropout_df['dataset'].unique()
    distinct_dropout = dropout_df['dropout'].unique()

    for dataset in distinct_datasets:
        filtered_dropout_df = dropout_df[(dropout_df['dataset'] == dataset)]
        distinct_methods = dropout_df['method'].unique()
        for method in distinct_methods:
            filtered_dropout_df = filtered_dropout_df[(dropout_df['method'] == method)]
            filtered_dropout_df = filtered_dropout_df.reset_index()
            best_ind = filtered_dropout_df[metric_column].idxmax()
            best_filtered_dropout_df = filtered_dropout_df.iloc[[best_ind]]                   
             
            run_save_path = best_filtered_dropout_df['run save path'].values[0]
            
            if metric_column.startswith('valid'):
                dataset_split = 'Validation'
                path = run_save_path + '/best_valid_perf.pkl'
            elif metric_column.startswith('test'):
                dataset_split = 'Test'
                path = run_save_path + '/best_test_perf.pkl'
            else:
                print('Invalid metric column')
                return

            #load validation and test performance files 
            with open(path, "rb") as input_file:
                best_perf = pickle.load(input_file)

            y_pred = best_perf['pred']
            y_true = best_perf['gt']
            method_title = 'MCD - ' + str(round(1-best_filtered_dropout_df['dropout'].values[0], 2))
            title =  dataset_split + ' Confusion Matrix for {0} on {1}'.format(method_title, dataset.title())

            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])

            # Plot the confusion matrix
            fig, ax = plt.subplots()
            disp.plot(cmap=plt.cm.Blues, ax=ax)

            # Set custom axis labels
            ax.set_xlabel('Predicted')  # Custom label for the x-axis
            ax.set_ylabel('Ground Truth') # Custom label for the y-axis
            plt.title(title)
            plt.show()


            # ROC Curve
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
            plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve for {0} on {1}'.format(method_title, dataset.title()))
            plt.legend(loc='lower right')
            plt.grid()
            plt.show()

            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_true, y_pred)

            plt.figure()
            plt.plot(recall, precision, color='blue', lw=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('PR Curve for {0} on {1}'.format(method_title, dataset.title()))
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.grid()
            plt.show()

def report_best_uncertainty_model_comparison(metric_column, metric_name, is_metric_percentage):
    dp_uncertainty_df = pd.read_csv('./experiment1/dropout_uncertainty/perf_dropout_results.csv')      
    dropout_df = pd.read_csv('./experiment1/dropout/perf_dropout_results.csv')

    distinct_datasets = dropout_df['dataset'].unique()

    for dataset in distinct_datasets:
        filtered_dropout_uncertainty_dataset_df = dp_uncertainty_df[(dp_uncertainty_df['dataset'] == dataset)]
        filtered_dropout_df = dropout_df[(dropout_df['dataset'] == dataset)]
        distinct_methods = filtered_dropout_df['method'].unique()

        for method in distinct_methods:
            filtered_dropout_method_df = filtered_dropout_df[(filtered_dropout_df['method'] == method)]
            filtered_dropout_df = filtered_dropout_df.reset_index()
            best_ind = filtered_dropout_df[metric_column].idxmax()
            best_filtered_dropout_df = filtered_dropout_df.iloc[[best_ind]]
            dropout = best_filtered_dropout_df['dropout'].values[0]

            filtered_dropout_uncertainty_method_df = filtered_dropout_uncertainty_dataset_df[(filtered_dropout_uncertainty_dataset_df['method'] == method) \
                & (filtered_dropout_uncertainty_dataset_df['dropout'] == dropout)]

            distinct_runs = filtered_dropout_uncertainty_method_df['run'].unique()
            df_runs = None

            for run in distinct_runs:
                filtered_dropout_uncertainty_run_df = filtered_dropout_uncertainty_method_df[(filtered_dropout_uncertainty_method_df['run'] == run)]
                filtered_dropout_uncertainty_grouped_df = filtered_dropout_uncertainty_run_df[[metric_column, 'run']].groupby(['run']).mean()
                # filtered_dropout_uncertainty_run_df = filtered_dropout_uncertainty_run_df.reset_index()
                # best_ind = filtered_dropout_uncertainty_run_df[metric_column].idxmax()
                # best_filtered_dropout_uncertainty_df = filtered_dropout_uncertainty_run_df.iloc[[best_ind]]
                
                df_compare_runs = pd.DataFrame({
                    "Run": [str(round(run, 2))], 
                    metric_name: [filtered_dropout_uncertainty_grouped_df[metric_column].values[0]]
                    })
                if df_runs is None:
                    df_runs = df_compare_runs
                else:
                    df_runs = pd.concat([df_runs, df_compare_runs])   

            runs = df_runs['Run']
            metric_values = df_runs[metric_name]

            # Create a bar chart
            plt.figure(figsize=(6, 3))
            bars = plt.bar(runs, metric_values, color=['#A3C1DA', '#007BFF'], width=0.4)
            avg_metric_value = df_runs[metric_name].mean()
            min_metric_value = df_runs[metric_name].min()
            max_metric_value = df_runs[metric_name].max()
            ylim_min = min_metric_value - (avg_metric_value * 0.25)
            ylim_max = max_metric_value + (avg_metric_value * 0.15)

            for bar in bars:
                yval = bar.get_height()  # Get the height of the bar
                text = round(((yval - max_metric_value) / max_metric_value) * 100, 2)
                if text < 0:
                    text = '-' + str(text) + '%'
                else:
                    text = '+' + str(text) + '%'
                plt.text(bar.get_x() + bar.get_width() / 2, yval, 
                text, ha='center', va='bottom')  # Add labels to the top of the bars 

                height = bar.get_height()
                # Positioning the text inside the bar
                if is_metric_percentage:
                    yval_text = round(yval, 2)
                else:
                    yval_text = round(yval, 3)

                plt.text(
                    bar.get_x() + bar.get_width() / 2,  # X position (middle of the bar)
                    yval - ((yval - ylim_min) / 2),  # Y position (half the height of the bar)
                    str(yval_text),  # The value to display
                    fontsize=8,  # Font size (adjust this value)
                    ha='center',  # Horizontal alignment
                    va='bottom',  # Vertical alignment
                ) 

            plt.axhline(y=max_metric_value, color='r', linestyle='--', linewidth=0.75, label='Top Run {0} %'.format(metric_column))

            # Adding labels and title
            plt.xlabel('Runs')
            plt.ylabel(metric_name)
            plt.title('Comparison of Thresholding Model Performance on {0}'.format(dataset.title()))

            plt.xticks(fontsize=6)

            plt.ylim(ylim_min, ylim_max)  # Set y-axis limits

            # Show the plot
            plt.grid(axis='y', alpha=0.75)
            plt.show()

def report_best_model_comparison_test(metric_column, metric_name, is_metric_percentage):
    replication_df = pd.read_csv('./replication/perf_results.csv')
    dropout_df = pd.read_csv('./experiment1/dropout/perf_dropout_results.csv')
    dp_uncertainty_df = pd.read_csv('./experiment1/dropout_uncertainty/perf_dropout_results.csv')
    
    #dropout_df['dropout'] = 1 - dropout_df['dropout']
    dropout_df = dropout_df.sort_values(by='dropout', ascending=True)
    distinct_datasets = dropout_df['dataset'].unique()
    distinct_dropout = dropout_df['dropout'].unique()

    for dataset in distinct_datasets:
        filtered_replication_dataset_df = replication_df[(replication_df['dataset'] == dataset)]
        filtered_dropout_df = dropout_df[(dropout_df['dataset'] == dataset)]
        distinct_methods = filtered_dropout_df['method'].unique()

        replication_grouped_max = filtered_replication_dataset_df[['method', 'dataset', metric_column]].groupby(['method', 'dataset']).max() \
                                    .sort_values(by=metric_column, ascending=True) \
                                    .reset_index()

        df_runs = pd.DataFrame({
        "Model": replication_grouped_max['method'], 
        metric_name: replication_grouped_max[metric_column]
        })

        for method in distinct_methods:
            filtered_replication_df = filtered_replication_dataset_df[(filtered_replication_dataset_df['method'] == method)]        
            filtered_replication_df = filtered_replication_df.reset_index()
            best_ind = filtered_replication_df[metric_column].idxmax()
            best_filtered_replication_df = filtered_replication_df.iloc[[best_ind]]

            filtered_dropout_method_df = filtered_dropout_df[(filtered_dropout_df['method'] == method)]
            filtered_dropout_method_df = filtered_dropout_method_df.reset_index()
            best_ind = filtered_dropout_method_df['valid acc'].idxmax()
            best_filtered_dropout_df = filtered_dropout_method_df.iloc[[best_ind]]

            distinct_dropout = best_filtered_dropout_df['dropout'].unique()
            for dropout in distinct_dropout:
                dp_uncertainty_df_filtered = dp_uncertainty_df[(dp_uncertainty_df['dropout'] == dropout) & \
                    (dp_uncertainty_df['dataset'] == dataset) & (dp_uncertainty_df['method'] == method) & \
                    (dp_uncertainty_df['prob measure'] == 'entr')]
                    
                mean_metric = dp_uncertainty_df_filtered[['dropout', 'dataset', 'method', 'prob measure', metric_column]] \
                    .groupby(['dropout', 'dataset', 'method', 'prob measure']).mean()[metric_column].values[0]

                df_compare_runs = pd.DataFrame({
                    "Model": ["Adv-ALSTM MCD " + str(round(1 - dropout, 2)) + ' w/ Thresholding'], 
                    metric_name: [mean_metric]
                    })
                if df_runs is None:
                    df_runs = df_compare_runs
                else:
                    df_runs = pd.concat([df_runs, df_compare_runs])   

            models = df_runs['Model']
            metric_values = df_runs[metric_name]

            # Create a bar chart
            plt.figure(figsize=(6, 3))
            bars = plt.bar(models, metric_values, color=['#A3C1DA', '#007BFF'], width=0.4)
            replication_metric = best_filtered_replication_df[metric_column].values[0]

            avg_metric_value = df_runs[metric_name].mean()
            min_metric_value = df_runs[metric_name].min()
            max_metric_value = df_runs[metric_name].max()
            ylim_min = min_metric_value - (avg_metric_value * 0.25)
            ylim_max = max_metric_value + (avg_metric_value * 0.15)

            for bar in bars:
                yval = bar.get_height()  # Get the height of the bar
                text = round(((yval - replication_metric) / replication_metric) * 100, 2)
                if text < 0:
                    text = '-' + str(text) + '%'
                else:
                    text = '+' + str(text) + '%'
                plt.text(bar.get_x() + bar.get_width() / 2, yval, 
                text, ha='center', va='bottom')  # Add labels to the top of the bars
                
                               # Positioning the text inside the bar
                if is_metric_percentage:
                    yval_text = round(yval, 2)
                else:
                    yval_text = round(yval, 3)

                plt.text(
                    bar.get_x() + bar.get_width() / 2,  # X position (middle of the bar)
                    yval - ((yval - ylim_min) / 2),  # Y position (half the height of the bar)
                    str(yval_text),  # The value to display
                    fontsize=6,  # Font size (adjust this value)
                    ha='center',  # Horizontal alignment
                    va='bottom',  # Vertical alignment
                ) 
            plt.axhline(y=best_filtered_replication_df[metric_column].values[0], color='r', linestyle='--', linewidth=0.75, label='Benchmark %')

            # Adding labels and title
            plt.xlabel('Models')
            plt.ylabel(metric_name)
            plt.title('Comparison of Model Performance on {0}'.format(dataset.title()))

            plt.xticks(fontsize=6)

            plt.ylim(ylim_min, ylim_max)  # Set y-axis limits

            # Show the plot
            plt.grid(axis='y', alpha=0.75)
            plt.show()
            
def report_best_dropout_model_comparison_test(metric_column, metric_name, is_metric_percentage):
    replication_df = pd.read_csv('./replication/perf_results.csv')
    dropout_df = pd.read_csv('./experiment1/dropout/perf_dropout_results.csv')
    dropout_df['dropout'] = 1 - dropout_df['dropout']
    dropout_df = dropout_df.sort_values(by='dropout', ascending=True)
    distinct_datasets = dropout_df['dataset'].unique()
    distinct_dropout = dropout_df['dropout'].unique()

    for dataset in distinct_datasets:
        filtered_replication_dataset_df = replication_df[(replication_df['dataset'] == dataset)]
        filtered_dropout_df = dropout_df[(dropout_df['dataset'] == dataset)]
        distinct_methods = filtered_dropout_df['method'].unique()

        for method in distinct_methods:
            filtered_replication_df = filtered_replication_dataset_df[(filtered_replication_dataset_df['method'] == method)]
            filtered_replication_df = filtered_replication_df.reset_index()
            best_ind = filtered_replication_df[metric_column].idxmax()
            best_filtered_replication_df = filtered_replication_df.iloc[[best_ind]]
            df_runs = pd.DataFrame({
            "Model": [best_filtered_replication_df['method'].values[0]], 
            metric_name: [best_filtered_replication_df[metric_column].values[0]]
            })

            filtered_dropout_method_df = filtered_dropout_df[(filtered_dropout_df['method'] == method)]
            distinct_dropout = filtered_dropout_method_df['dropout'].unique()
            for dropout in distinct_dropout:
                filtered_dropout_df = filtered_dropout_method_df[(filtered_dropout_method_df['dropout'] == dropout)]
                filtered_dropout_df = filtered_dropout_df.reset_index()
                best_ind = filtered_dropout_df[metric_column].idxmax()
                best_filtered_dropout_df = filtered_dropout_df.iloc[[best_ind]]

                df_compare_runs = pd.DataFrame({
                    "Model": ["MCD " + str(round(best_filtered_dropout_df['dropout'].values[0], 2))], 
                    metric_name: [best_filtered_dropout_df[metric_column].values[0]]
                    })
                if df_runs is None:
                    df_runs = df_compare_runs
                else:
                    df_runs = pd.concat([df_runs, df_compare_runs])   

            models = df_runs['Model']
            metric_values = df_runs[metric_name]

            # Create a bar chart
            plt.figure(figsize=(6, 3))
            bars = plt.bar(models, metric_values, color=['#A3C1DA', '#007BFF'], width=0.4)
            replication_metric = best_filtered_replication_df[metric_column].values[0]

            avg_metric_value = df_runs[metric_name].mean()
            min_metric_value = df_runs[metric_name].min()
            max_metric_value = df_runs[metric_name].max()
            ylim_min = min_metric_value - (avg_metric_value * 0.25)
            ylim_max = max_metric_value + (avg_metric_value * 0.15)

            for bar in bars:
                yval = bar.get_height()  # Get the height of the bar
                text = round(((yval - replication_metric) / replication_metric) * 100, 2)
                if text < 0:
                    text = '-' + str(text) + '%'
                else:
                    text = '+' + str(text) + '%'
                plt.text(bar.get_x() + bar.get_width() / 2, yval, 
                text, ha='center', va='bottom')  # Add labels to the top of the bars
                
                               # Positioning the text inside the bar
                if is_metric_percentage:
                    yval_text = round(yval, 2)
                else:
                    yval_text = round(yval, 3)

                plt.text(
                    bar.get_x() + bar.get_width() / 2,  # X position (middle of the bar)
                    yval - ((yval - ylim_min) / 2),  # Y position (half the height of the bar)
                    str(yval_text),  # The value to display
                    fontsize=6,  # Font size (adjust this value)
                    ha='center',  # Horizontal alignment
                    va='bottom',  # Vertical alignment
                ) 
            plt.axhline(y=best_filtered_replication_df[metric_column].values[0], color='r', linestyle='--', linewidth=0.75, label='Benchmark %')

            # Adding labels and title
            plt.xlabel('Models')
            plt.ylabel(metric_name)
            plt.title('Comparison of Model Performance on {0}'.format(dataset.title()))

            plt.xticks(fontsize=6)

            plt.ylim(ylim_min, ylim_max)  # Set y-axis limits

            # Show the plot
            plt.grid(axis='y', alpha=0.75)
            plt.show()

def report_best_dropout_run_binary_valid_graphs(metric_column):
    dropout_df = pd.read_csv('./experiment1/dropout/perf_dropout_results.csv')
    distinct_datasets = dropout_df['dataset'].unique()
    distinct_dropout = dropout_df['dropout'].unique()

    for dataset in distinct_datasets:
        filtered_dropout_df = dropout_df[(dropout_df['dataset'] == dataset)]
        distinct_methods = dropout_df['method'].unique()
        for method in distinct_methods:
            filtered_dropout_df = filtered_dropout_df[(dropout_df['method'] == method)]
            filtered_dropout_df = filtered_dropout_df.reset_index()
            best_ind = filtered_dropout_df[metric_column].idxmax()
            best_filtered_dropout_df = filtered_dropout_df.iloc[[best_ind]]                   
             
            run_save_path = best_filtered_dropout_df['run save path'].values[0]
            
            if metric_column.startswith('valid'):
                dataset_split = 'Validation'
                path = run_save_path + '/best_valid_perf.pkl'
            elif metric_column.startswith('test'):
                dataset_split = 'Test'
                path = run_save_path + '/best_test_perf.pkl'
            else:
                print('Invalid metric column')
                return

            #load validation and test performance files 
            with open(path, "rb") as input_file:
                best_perf = pickle.load(input_file)

            y_pred = best_perf['pred']
            y_true = best_perf['gt']
            method_title = 'MCD - ' + str(round(1-best_filtered_dropout_df['dropout'].values[0], 2))
            title =  dataset_split + ' Confusion Matrix for {0} on {1}'.format(method_title, dataset.title())

            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])

            # Plot the confusion matrix
            fig, ax = plt.subplots()
            disp.plot(cmap=plt.cm.Blues, ax=ax)

            # Set custom axis labels
            ax.set_xlabel('Predicted')  # Custom label for the x-axis
            ax.set_ylabel('Ground Truth') # Custom label for the y-axis
            plt.title(title)
            plt.show()
            
            for mi, m in enumerate(best_valid_perf['prob_arr']):
                val_pre_prob = best_valid_perf['prob_arr'][mi]['val']
                tes_pre_prob = best_test_perf['prob_arr'][mi]['val']

                # ROC Curve
                fpr, tpr, _ = roc_curve(y_pred, val_pre_prob)
                roc_auc = auc(fpr, tpr)

                plt.figure()
                plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
                plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.0])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve for {0} on {1}'.format(method_title, dataset.title()))
                plt.legend(loc='lower right')
                plt.grid()
                plt.show()

                # Precision-Recall Curve
                precision, recall, _ = precision_recall_curve(y_pred, val_pre_prob)

                plt.figure()
                plt.plot(recall, precision, color='blue', lw=2)
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('PR Curve for {0} on {1}'.format(method_title, dataset.title()))
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.0])
                plt.grid()
                plt.show()

def report_pr_curves(metric_column, metric_name, metric_title_name, colors):
    dp_df = pd.read_csv('./experiment1/dropout/perf_dropout_results.csv')
    dp_uncertainty_df = pd.read_csv('./experiment1/dropout_uncertainty/perf_dropout_results.csv')      

    distinct_datasets = dp_df['dataset'].unique()
    distinct_dropout = dp_df['dropout'].unique()

    for dataset in distinct_datasets:
        filtered_dropout_df = dp_df[(dp_df['dataset'] == dataset)]
        distinct_methods = dp_df['method'].unique()
        for method in distinct_methods:
            filtered_dropout_method_df = filtered_dropout_df[(filtered_dropout_df['method'] == method)]

            #get best experiment 1 dropout run by validation accuracy
            dp_df_filter = dp_df[(dp_df['dataset'] == dataset) & (dp_df['method'] == method)]
            dp_df_filter = dp_df_filter.reset_index()
            val_best_ind = dp_df_filter['valid acc'].idxmax()
            val_best_benchmark_model = dp_df_filter.iloc[[val_best_ind]]
            val_best_benchmark_model = val_best_benchmark_model.reset_index()
            val_best_ind = val_best_benchmark_model['valid mcc'].idxmax()
            val_best_benchmark_model = val_best_benchmark_model.iloc[val_best_ind]
            dropout = val_best_benchmark_model['dropout']
            run = int(val_best_benchmark_model['run'])

            filtered_dp_uncertainty_df = dp_uncertainty_df[(dp_uncertainty_df['dropout'] == dropout) & \
                 (dp_uncertainty_df['dataset'] == dataset) & (dp_uncertainty_df['method'] == method) & \
                 (dp_uncertainty_df['total valid predictions'] > 0)] \
                 .sort_values(by=metric_column, ascending=False)  
            filtered_dp_uncertainty_df = filtered_dp_uncertainty_df.reset_index()
            val_best_uncertainty_ind = filtered_dp_uncertainty_df['valid acc'].idxmax()
            val_best_uncertainty_benchmark_model = filtered_dp_uncertainty_df.iloc[[val_best_uncertainty_ind]]
            run2 = val_best_uncertainty_benchmark_model['run'].values[0]

            filtered_dp_uncertainty_df = filtered_dp_uncertainty_df.reset_index()
            val_best_uncertainty_ind = filtered_dp_uncertainty_df['valid mcc'].idxmax()
            val_best_uncertainty_benchmark_model = filtered_dp_uncertainty_df.iloc[[val_best_uncertainty_ind]]
            run3 = val_best_uncertainty_benchmark_model['run'].values[0]


            filtered_dp_uncertainty_run_df = dp_uncertainty_df[(dp_uncertainty_df['dropout'] == dropout) & \
                (dp_uncertainty_df['dataset'] == dataset) & (dp_uncertainty_df['method'] == method)]

            filtered_dp_uncertainty_run_df = filtered_dp_uncertainty_run_df[(filtered_dp_uncertainty_run_df['run'] == run) | \
                (filtered_dp_uncertainty_run_df['run'] == run2) | (filtered_dp_uncertainty_run_df['run'] == run3)] \
                .sort_values(by=metric_column, ascending=False)  

            distinct_runs = filtered_dp_uncertainty_run_df['run'].unique()

            fig, ax = plt.subplots()
            for ind, r in enumerate(distinct_runs):
                filtered_dp_run_df = filtered_dp_uncertainty_run_df[(filtered_dp_uncertainty_run_df['run'] == r)]
                filtered_dp_run_df = filtered_dp_run_df[['run', 'prob confidence threshold', 'valid ps', 'valid rs']].groupby(['run', 'prob confidence threshold']).mean() \
                    .sort_values(by='prob confidence threshold', ascending=True)
                
                # df_run = dropout_dict[ind].sort_values(by='dropout', ascending=True).reset_index()
                # df_run['dropout'] =  1-df_run['dropout']
                # best_run = dropout_dict[distinct_run.max()-1].sort_values(by='dropout', ascending=True).reset_index()
                # df_run['metric diff from best'] =  (df_run[metric_column] - best_run[metric_column]).round(4)
                # print(df_run[['dataset', 'dropout', 'run', 'valid acc', 'valid mcc', 'metric diff from best']])
                y = filtered_dp_run_df['valid ps']
                x = filtered_dp_run_df['valid rs']

                # Create a scatter plot
                ax.scatter(x, y, color=colors[ind])
                ax.plot(x, y, color=colors[ind], linestyle='-', linewidth=1, label='Run ' + str(int(r)))
            
                ax.fill_between(x, y, color=colors[ind], alpha=0.4)

                ax.set_xlabel("Probability Confidence Threshold")
                ax.set_ylabel(metric_name)

                # for index, x_point in enumerate(x):                
                
                #     if index + 1 < len(x):
                #         x_point_2 = x.iloc[index + 1]
                #         y_lower = y.min()
                #         y_upper = y.iloc[index]
                #         x_point = x.iloc[index]
                    
                #         plt.fill_between([x_point, x_point_2], y_lower, y_upper, color=colors[ind], alpha=0.4)


            if dataset == 'stocknet':
                dataset_title = 'Acl18'
            else:
                dataset_title = dataset.title()
            
            min_y = filtered_dp_uncertainty_run_df['valid ps'].min()
            max_y = filtered_dp_uncertainty_run_df['valid ps'].max()
            diff =  max_y - min_y
            top_y = max_y + (diff * 0.10)
            plt.ylim(bottom=min_y, top=top_y)

            # Adding labels and title
            #plt.ylabel(metric_name)
            #plt.xlabel('Probability Confidence Threshold')
            dropout = 1-dropout
            plt.title('{0} vs PCT for {1} Runs (Dropout {2}, Run {3})'.format(metric_title_name, dataset_title, str(math.floor(dropout * 100)/100.0), str(int(r))))
            plt.legend()

            # Show the plot
            plt.grid()
            plt.show()

def report_dropout_accumlative_returns(metric_column, metric_name, metric_title_name, colors):
    dp_df = pd.read_csv('./experiment1/dropout/perf_dropout_results.csv')
    dp_returns_val_df = pd.read_csv('./experiment2/dropout/run/dropout_pre_returns_val_results')      

    distinct_datasets = dp_df['dataset'].unique()
    distinct_dropout = dp_df['dropout'].unique()

    for dataset in distinct_datasets:
        filtered_dropout_df = dp_df[(dp_df['dataset'] == dataset)]
        distinct_methods = dp_df['method'].unique()
        for method in distinct_methods:
            filtered_dropout_method_df = filtered_dropout_df[(filtered_dropout_df['method'] == method)]

            #get best experiment 1 dropout run by validation accuracy
            dp_df_filter = dp_df[(dp_df['dataset'] == dataset) & (dp_df['method'] == method)]
            dp_df_filter = dp_df_filter.reset_index()
            val_best_ind = dp_df_filter['valid acc'].idxmax()
            val_best_benchmark_model = dp_df_filter.iloc[[val_best_ind]]
            val_best_benchmark_model = val_best_benchmark_model.reset_index()
            val_best_ind = val_best_benchmark_model['valid mcc'].idxmax()
            val_best_benchmark_model = val_best_benchmark_model.iloc[val_best_ind]
            dropout = val_best_benchmark_model['dropout']
            #run = int(val_best_benchmark_model['run'])

            filtered_dp_uncertainty_df = dp_uncertainty_df[(dp_uncertainty_df['dropout'] == dropout) & \
                 (dp_uncertainty_df['dataset'] == dataset) & (dp_uncertainty_df['method'] == method)] \
                 .sort_values(by=metric_column, ascending=False)  
            filtered_dp_uncertainty_df = filtered_dp_uncertainty_df.reset_index()
            val_best_uncertainty_ind = filtered_dp_uncertainty_df['valid acc'].idxmax()
            val_best_uncertainty_benchmark_model = filtered_dp_uncertainty_df.iloc[[val_best_uncertainty_ind]]
            run = val_best_uncertainty_benchmark_model['run'].values[0]

            filtered_dp_uncertainty_run_df = dp_uncertainty_df[(dp_uncertainty_df['dropout'] == dropout) & \
                (dp_uncertainty_df['dataset'] == dataset) & (dp_uncertainty_df['method'] == method)]

            filtered_dp_uncertainty_run_df = filtered_dp_uncertainty_run_df[(filtered_dp_uncertainty_run_df['run'] == run)] \
                 .sort_values(by=metric_column, ascending=False)  

            distinct_runs = filtered_dp_uncertainty_run_df['run'].unique()

            for r in distinct_runs:
                fig, ax = plt.subplots()
                filtered_dp_run_df = filtered_dp_uncertainty_df[(filtered_dp_uncertainty_df['run'] == r)]
                distinct_uncertainty_measures = filtered_dp_run_df \
                                    .sort_values(by='prob measure', ascending=True) \
                                    ['prob measure'] \
                                    .unique()

                for ind, prob_measure in enumerate(distinct_uncertainty_measures):
                    filtered_dp_prob_measures_df = filtered_dp_run_df[
                        (filtered_dp_run_df['prob measure'] == prob_measure)] \
                    .sort_values(by='prob confidence threshold', ascending=True)
                    
                    # df_run = dropout_dict[ind].sort_values(by='dropout', ascending=True).reset_index()
                    # df_run['dropout'] =  1-df_run['dropout']
                    # best_run = dropout_dict[distinct_run.max()-1].sort_values(by='dropout', ascending=True).reset_index()
                    # df_run['metric diff from best'] =  (df_run[metric_column] - best_run[metric_column]).round(4)
                    # print(df_run[['dataset', 'dropout', 'run', 'valid acc', 'valid mcc', 'metric diff from best']])
                    y = filtered_dp_prob_measures_df[metric_column]
                    x = filtered_dp_prob_measures_df['prob confidence threshold']

                    if prob_measure == 'entr':
                        prob_measure_title = 'Entropy'
                    elif prob_measure == 'var':
                        prob_measure_title = 'Variance'
                    elif prob_measure == 'std':
                        prob_measure_title = 'Standard Deviation'

                    # Create a scatter plot
                    ax.scatter(x, y, color=colors[ind])
                    ax.plot(x, y, color=colors[ind], linestyle='-', linewidth=1, label=prob_measure_title)
                
                    ax.fill_between(x, y, color=colors[ind], alpha=0.4)

                    ax.set_xlabel("Probability Confidence Threshold")
                    ax.set_ylabel(metric_name)

                    # for index, x_point in enumerate(x):                
                    
                    #     if index + 1 < len(x):
                    #         x_point_2 = x.iloc[index + 1]
                    #         y_lower = y.min()
                    #         y_upper = y.iloc[index]
                    #         x_point = x.iloc[index]
                        
                    #         plt.fill_between([x_point, x_point_2], y_lower, y_upper, color=colors[ind], alpha=0.4)


                if dataset == 'stocknet':
                    dataset_title = 'Acl18'
                else:
                    dataset_title = dataset.title()
                
                min_y = filtered_dp_run_df[metric_column].min()
                max_y = filtered_dp_run_df[metric_column].max()
                diff =  max_y - min_y
                top_y = max_y + (diff * 0.10)
                plt.ylim(bottom=min_y, top=top_y)

                # Adding labels and title
                #plt.ylabel(metric_name)
                #plt.xlabel('Probability Confidence Threshold')
                dropout = 1-dropout
                plt.title('{0} vs PCT for {1} Runs (Dropout {2}, Run {3})'.format(metric_title_name, dataset_title, str(math.floor(dropout * 100)/100.0), str(int(r))))
                plt.legend()

                # Show the plot
                plt.grid()
                plt.show()

def report_dropout_uncertainty(metric_column, metric_name, metric_title_name, colors):
    dp_df = pd.read_csv('./experiment1/dropout/perf_dropout_results.csv')
    dp_uncertainty_df = pd.read_csv('./experiment1/dropout_uncertainty/perf_dropout_results.csv')      

    distinct_datasets = dp_df['dataset'].unique()
    distinct_dropout = dp_df['dropout'].unique()

    for dataset in distinct_datasets:
        filtered_dropout_df = dp_df[(dp_df['dataset'] == dataset)]
        distinct_methods = dp_df['method'].unique()
        for method in distinct_methods:
            filtered_dropout_method_df = filtered_dropout_df[(filtered_dropout_df['method'] == method)]

            #get best experiment 1 dropout run by validation accuracy
            dp_df_filter = dp_df[(dp_df['dataset'] == dataset) & (dp_df['method'] == method)]
            dp_df_filter = dp_df_filter.reset_index()
            val_best_ind = dp_df_filter['valid acc'].idxmax()
            val_best_benchmark_model = dp_df_filter.iloc[[val_best_ind]]
            val_best_benchmark_model = val_best_benchmark_model.reset_index()
            val_best_ind = val_best_benchmark_model['valid mcc'].idxmax()
            val_best_benchmark_model = val_best_benchmark_model.iloc[val_best_ind]
            dropout = val_best_benchmark_model['dropout']
            #run = int(val_best_benchmark_model['run'])

            filtered_dp_uncertainty_df = dp_uncertainty_df[(dp_uncertainty_df['dropout'] == dropout) & \
                 (dp_uncertainty_df['dataset'] == dataset) & (dp_uncertainty_df['method'] == method)] \
                 .sort_values(by=metric_column, ascending=False)  
            filtered_dp_uncertainty_df = filtered_dp_uncertainty_df.reset_index()
            val_best_uncertainty_ind = filtered_dp_uncertainty_df['valid acc'].idxmax()
            val_best_uncertainty_benchmark_model = filtered_dp_uncertainty_df.iloc[[val_best_uncertainty_ind]]
            run = val_best_uncertainty_benchmark_model['run'].values[0]

            filtered_dp_uncertainty_run_df = dp_uncertainty_df[(dp_uncertainty_df['dropout'] == dropout) & \
                (dp_uncertainty_df['dataset'] == dataset) & (dp_uncertainty_df['method'] == method)]

            filtered_dp_uncertainty_run_df = filtered_dp_uncertainty_run_df[(filtered_dp_uncertainty_run_df['run'] == run)] \
                 .sort_values(by=metric_column, ascending=False)  

            distinct_runs = filtered_dp_uncertainty_run_df['run'].unique()

            for r in distinct_runs:
                fig, ax = plt.subplots()
                filtered_dp_run_df = filtered_dp_uncertainty_df[(filtered_dp_uncertainty_df['run'] == r)]
                distinct_uncertainty_measures = filtered_dp_run_df \
                                    .sort_values(by='prob measure', ascending=True) \
                                    ['prob measure'] \
                                    .unique()

                for ind, prob_measure in enumerate(distinct_uncertainty_measures):
                    filtered_dp_prob_measures_df = filtered_dp_run_df[
                        (filtered_dp_run_df['prob measure'] == prob_measure)] \
                    .sort_values(by='prob confidence threshold', ascending=True)
                    
                    # df_run = dropout_dict[ind].sort_values(by='dropout', ascending=True).reset_index()
                    # df_run['dropout'] =  1-df_run['dropout']
                    # best_run = dropout_dict[distinct_run.max()-1].sort_values(by='dropout', ascending=True).reset_index()
                    # df_run['metric diff from best'] =  (df_run[metric_column] - best_run[metric_column]).round(4)
                    # print(df_run[['dataset', 'dropout', 'run', 'valid acc', 'valid mcc', 'metric diff from best']])
                    y = filtered_dp_prob_measures_df[metric_column]
                    x = filtered_dp_prob_measures_df['prob confidence threshold']

                    if prob_measure == 'entr':
                        prob_measure_title = 'Entropy'
                    elif prob_measure == 'var':
                        prob_measure_title = 'Variance'
                    elif prob_measure == 'std':
                        prob_measure_title = 'Standard Deviation'

                    # Create a scatter plot
                    ax.scatter(x, y, color=colors[ind])
                    ax.plot(x, y, color=colors[ind], linestyle='-', linewidth=1, label=prob_measure_title)
                
                    ax.fill_between(x, y, color=colors[ind], alpha=0.4)

                    ax.set_xlabel("Probability Confidence Threshold")
                    ax.set_ylabel(metric_name)

                    # for index, x_point in enumerate(x):                
                    
                    #     if index + 1 < len(x):
                    #         x_point_2 = x.iloc[index + 1]
                    #         y_lower = y.min()
                    #         y_upper = y.iloc[index]
                    #         x_point = x.iloc[index]
                        
                    #         plt.fill_between([x_point, x_point_2], y_lower, y_upper, color=colors[ind], alpha=0.4)


                if dataset == 'stocknet':
                    dataset_title = 'Acl18'
                else:
                    dataset_title = dataset.title()
                
                min_y = filtered_dp_run_df[metric_column].min()
                max_y = filtered_dp_run_df[metric_column].max()
                diff =  max_y - min_y
                top_y = max_y + (diff * 0.10)
                plt.ylim(bottom=min_y, top=top_y)

                # Adding labels and title
                #plt.ylabel(metric_name)
                #plt.xlabel('Probability Confidence Threshold')
                dropout = 1-dropout
                plt.title('{0} vs PCT for {1} Runs (Dropout {2}, Run {3})'.format(metric_title_name, dataset_title, str(math.floor(dropout * 100)/100.0), str(int(r))))
                plt.legend()

                # Show the plot
                plt.grid()
                plt.show()

def report_uncertainty_heatmap(metric_column, metric_name):
    dp_df = pd.read_csv('./experiment1/dropout/perf_dropout_results.csv')     
    dp_uncertainty_df = pd.read_csv('./experiment1/dropout_uncertainty/perf_dropout_results.csv')      

    distinct_datasets = dp_df['dataset'].unique()
    distinct_methods = dp_df['method'].unique()

    for dataset in distinct_datasets:
        for method in distinct_methods:  
            #get best experiment 1 dropout run by validation accuracy
            dp_df_filter = dp_df[(dp_df['dataset'] == dataset) & (dp_df['method'] == method)]
            dp_df_filter = dp_df_filter.reset_index()
            val_best_ind = dp_df_filter['valid acc'].idxmax()
            val_best_benchmark_model = dp_df_filter.iloc[[val_best_ind]]
            val_best_benchmark_model = val_best_benchmark_model.reset_index()
            val_best_ind = val_best_benchmark_model['valid mcc'].idxmax()
            val_best_benchmark_model = val_best_benchmark_model.iloc[val_best_ind]

            dropout = val_best_benchmark_model['dropout']

            #get best experiment 1 dropout run by validation accuracy
            best_benchmark_dp_uncertainty_model = dp_uncertainty_df[(dp_uncertainty_df['dropout'] == dropout) & \
                (dp_uncertainty_df['dataset'] == dataset) & \
                (dp_uncertainty_df['method'] == method)]    
            filtered_dp_run_df = best_benchmark_dp_uncertainty_model[['run', 'valid acc', 'run save path']].groupby(['run']).mean()
            filtered_dp_run_df = filtered_dp_run_df.reset_index()
            val_best_ind = filtered_dp_run_df['valid acc'].idxmax()
            val_best_benchmark_uncertainty_model = filtered_dp_run_df.iloc[[val_best_ind]]
            run = val_best_benchmark_uncertainty_model['run'].values[0]

        
           #load the validation and test mappings
            mappings_save_path = './tmp/' + dataset
            with open(mappings_save_path + '/val_mappings.pkl', 'rb') as val_mappings_file:
                val_mappings = pickle.load(val_mappings_file)
            with open(mappings_save_path + '/tes_mappings.pkl', 'rb') as tes_mappings_file:
                tes_mappings = pickle.load(tes_mappings_file)

            run_save_path = best_benchmark_dp_uncertainty_model[(dp_uncertainty_df['run'] == run)]['run save path'].values[0] 

            #load validation and test performance files
            with open(run_save_path + '/best_test_perf.pkl', "rb") as input_file:
                best_test_perf = pickle.load(input_file)
            with open(run_save_path + '/best_valid_perf.pkl', "rb") as input_file:
                best_valid_perf = pickle.load(input_file)

            last_date = None
            last_date_index = -1
            stock_dic = {}
            for m in val_mappings:
                date = m['date']
                if date != last_date:
                    last_date_index = last_date_index + 1

                m['date_index'] = last_date_index

                if m['ticker_filename'] not in stock_dic:
                    stock_dic[m['ticker_filename']] = len(stock_dic)
    
                m['stock_index'] = stock_dic[m['ticker_filename']]
                last_date = date

            
            mappings_df = pd.DataFrame(val_mappings)

            for m in best_valid_perf['prob_arr']:
                prob_measure = m['measure']
                prob_val = m['val']

                prob_val_df = pd.DataFrame(prob_val, columns=['prob_val'])
                merge_df = pd.merge(prob_val_df, \
                mappings_df, left_index=True, right_index=True, suffixes=('_df1', '_df2'))
                
                merge_df = merge_df \
                .sort_values(by=['date_index', 'stock_index'], ascending=True) \
                .reset_index()

                distinct_date_indexes = merge_df['date_index'].unique()

                data_array = []
                for index_row in distinct_date_indexes:
                    merge_df_filter = merge_df[(merge_df['date_index'] == index_row)] \
                    .sort_values(by='stock_index', ascending=True) \
                    .reset_index()
                    
                    a = np.empty(len(stock_dic), float)
                    a.fill(np.nan)
                    data_array.append(a)

                    # shape=(len(distinct_date_indexes), len(stock_dic)

                    for index, row in merge_df_filter.iterrows():
                        data_array[index_row][index] = row['prob_val']

                data_array_np = np.array(data_array, ndmin=2)
                # Create the heatmap using imshow
                plt.imshow(data_array_np, cmap='YlOrRd', interpolation='nearest')
                # Add contour lines for high activity areas
                # contours = plt.contour(data_array_np, levels=[6, 8], colors='red', linewidths=2)


                if prob_measure == 'entr':
                    prob_measure_title = 'Entropy'
                elif prob_measure == 'var':
                    prob_measure_title = 'Variance'
                elif prob_measure == 'std':
                    prob_measure_title = 'Standard Deviation'

                if dataset == 'stocknet':
                    dataset_title = 'Acl18'
                else:
                    dataset_title = dataset.title()

                # Add a colorbar to show the scale of values
                plt.colorbar(label=prob_measure_title)

                dropout_text = 1 - val_best_benchmark_model['dropout']

                # Add labels and title
                plt.title(dataset_title + ' Heatmap of Uncertainty (Run: ' + str(run) + ')')
                plt.xlabel('Stock Index')
                plt.ylabel('Trading Day Index')

                # Show the plot
                plt.show()

def report_compare_ticker_replication(metric_column, metric_name):
    replication_tickers_df = pd.read_csv('./experiment2/replication/replication_pre_tes_ticker_returns_results.csv')
    dropout_tickers_df = pd.read_csv('./experiment2/dropout/ticker/dropout_pre_tes_ticker_returns_results.csv')
    dp_uncertainty_df = pd.read_csv('./experiment1/dropout_uncertainty/perf_dropout_results.csv')      
    dp_df = pd.read_csv('./experiment1/dropout/perf_dropout_results.csv')      
    benchmark_df = pd.read_csv('./experiment2/replication/replication_pre_returns_results.csv')

    distinct_datasets = dp_df['dataset'].unique()
    distinct_methods = dp_df['method'].unique()

    for dataset in distinct_datasets:
        for method in distinct_methods:  
            dataset = 'kdd17'           

            #get best experiment 2 replication run by total validation returns
            best_benchmark_model = benchmark_df[(benchmark_df['dataset'] == dataset) & (benchmark_df['method'] == method)]
            best_benchmark_model = best_benchmark_model.reset_index()
            #val_best_ind = best_benchmark_model['total val log return'].idxmax()
            val_best_ind = best_benchmark_model['total val return'].idxmax()
            val_best_benchmark_model = best_benchmark_model.iloc[val_best_ind]
            val_best_benchmark_model_run = int(val_best_benchmark_model['run'])

            #get best experiment 2 replication run by total test returns
            best_benchmark_model = best_benchmark_model.reset_index()
            #tes_best_ind = best_benchmark_model['total tes log return'].idxmax()
            tes_best_ind = best_benchmark_model['total tes return'].idxmax()
            tes_best_benchmark_model = best_benchmark_model.iloc[tes_best_ind]
            tes_best_benchmark_model_run = int(tes_best_benchmark_model['run'])    
            #get best experiment 1 dropout run by validation accuracy
            dp_df_filter = dp_df[(dp_df['dataset'] == dataset) & (dp_df['method'] == method)]
            dp_df_filter = dp_df_filter.reset_index()
            val_best_ind = dp_df_filter['valid acc'].idxmax()
            val_best_benchmark_model = dp_df_filter.iloc[[val_best_ind]]
            val_best_benchmark_model = val_best_benchmark_model.reset_index()
            val_best_ind = val_best_benchmark_model['test acc'].idxmax()
            val_best_benchmark_model = val_best_benchmark_model.iloc[val_best_ind]

            #get best experiment 1 dropout run by validation accuracy
            best_benchmark_dp_uncertainty_model = dp_uncertainty_df[(dp_uncertainty_df['run'] == val_best_benchmark_model['run']) & (dp_uncertainty_df['total valid predictions'] > 10) & (dp_uncertainty_df['dropout'] == val_best_benchmark_model['dropout']) & (dp_uncertainty_df['dataset'] == val_best_benchmark_model['dataset']) & (dp_uncertainty_df['method'] == val_best_benchmark_model['method'])]
            best_benchmark_dp_uncertainty_model = best_benchmark_dp_uncertainty_model.reset_index()
            val_best_ind = best_benchmark_dp_uncertainty_model['valid acc'].idxmax()
            val_best_benchmark_uncertainty_model = best_benchmark_dp_uncertainty_model.iloc[[val_best_ind]]
            val_best_benchmark_uncertainty_model = val_best_benchmark_uncertainty_model.reset_index()
            val_best_ind = val_best_benchmark_uncertainty_model['test acc'].idxmax()
            val_uncertainty_best_benchmark_model = val_best_benchmark_uncertainty_model.iloc[val_best_ind]
            
            dropout_tickers_df = dropout_tickers_df[
                (dropout_tickers_df['run'] == val_uncertainty_best_benchmark_model['run']) & \
                (dropout_tickers_df['prob measure'] == val_uncertainty_best_benchmark_model['prob measure']) & \
                #(dropout_tickers_df['prob'] == val_uncertainty_best_benchmark_model['prob confidence threshold']) & \
                (dropout_tickers_df['prob'] == 0.1) & \
                (dropout_tickers_df['dropout'] == val_uncertainty_best_benchmark_model['dropout']) & \
                (dropout_tickers_df['dataset'] == val_uncertainty_best_benchmark_model['dataset']) & \
                (dropout_tickers_df['method'] == val_uncertainty_best_benchmark_model['method'])]
            tickers_replication_df = replication_tickers_df[
                (replication_tickers_df['dataset'] == val_uncertainty_best_benchmark_model['dataset']) & \
                (replication_tickers_df['method'] == val_uncertainty_best_benchmark_model['method']) & \
                (replication_tickers_df['run'] == tes_best_benchmark_model_run)]

            dropout_tickers_df = dropout_tickers_df \
            .sort_values(by='total return', ascending=False) \
            .reset_index()
            
            tickers_replication_df = tickers_replication_df \
            .sort_values(by='total return', ascending=False) \
            .reset_index() \
            .tail(25)

            merge_df = pd.merge(tickers_replication_df, \
            dropout_tickers_df, on='ticker filename', how='inner')
            sum_dropout = merge_df['total return_y'].sum()
            sum_replication = tickers_replication_df['total return'].sum()           

def report_dropout_grouped_runs(metric_column, metric_name, metric_title_name, convert_from_percentage):
    dropout_grouped_results_df = pd.read_csv('./experiment1/dropout/perf_dropout_grouped_results.csv')
    dropout_df = pd.read_csv('./experiment1/dropout/perf_dropout_results.csv')
    distinct_datasets = dropout_grouped_results_df['dataset'].unique()
    distinct_dropout = dropout_grouped_results_df['dropout'].unique()

    if (convert_from_percentage):
        dropout_grouped_results_df['avg ' + metric_column] = dropout_grouped_results_df['avg ' + metric_column] / 100
        dropout_df[metric_column] = dropout_df[metric_column] / 100

    for dataset in distinct_datasets:
        filtered_dropout_grouped_df = dropout_grouped_results_df[(dropout_grouped_results_df['dataset'] == dataset)]
        filtered_dropout_df = dropout_df[(dropout_df['dataset'] == dataset)]
        distinct_methods = filtered_dropout_grouped_df['method'].unique()

        for method in distinct_methods:
            filtered_dropout_grouped_df = filtered_dropout_grouped_df[(filtered_dropout_grouped_df['method'] == method)]
            filtered_dropout_grouped_df['std ' + metric_column] = None
            filtered_dropout_df = dropout_df[(dropout_df['method'] == method)]
        for dropout in distinct_dropout:
            filtered_dropout_subset_df = filtered_dropout_df[(filtered_dropout_df['dropout'] == dropout)]
            std_value = filtered_dropout_subset_df[metric_column].std()
        
            # Check if std_value is valid before assigning
            if not np.isnan(std_value):
                filtered_dropout_grouped_df.loc[filtered_dropout_grouped_df['dropout'] == dropout, 'std ' + metric_column] = std_value
        
        filtered_dropout_grouped_df = filtered_dropout_grouped_df.sort_values(by='dropout', ascending=True).reset_index()
        y = filtered_dropout_grouped_df['avg ' + metric_column]
        x = 1-filtered_dropout_grouped_df['dropout']
        z =  filtered_dropout_grouped_df['std ' + metric_column] 

        # Create a scatter plot
        plt.scatter(x, y, label='Average Run')
        plt.plot(x, y, color='#003366', linestyle='-', linewidth=1)
        plt.errorbar(x, y, yerr=z, fmt="o", ecolor='black', elinewidth=1, capsize=3, capthick=1, label='Standard Deviation')

        # Annotate each point with its value on the error bars
        prev = None
        for i in range(len(x)):
            text_y = y[i] + z[i] + 0.002
            text_x = x[i]
            if i == 1:
                text_x = text_x - 0.01
            elif i == 5:
                text_x = text_x + 0.02
            plt.text(text_x, text_y, f'{z[i]:.0e}', ha='center', fontsize=7, color='black')  # Above the upper error bar

        avg_metric_value = y.mean()
        min_metric_value = y.min() - z.max()
        max_metric_value = y.max() + z.max()
        if (max_metric_value < 0.2):
            scale = (max_metric_value- min_metric_value) * 1.15
        else:
            scale = (max_metric_value - min_metric_value) * 0.45

        plt.ylim(min_metric_value - scale, max_metric_value + scale)  # Set y-axis limits

        # Adding labels and title
        plt.ylabel(metric_name)
        plt.xlabel('Dropout')
        if dataset == 'stocknet':
            dataset_title = 'Acl18'
        else:
            dataset_title = dataset.title()
        plt.title('{0} vs Dropout for {1} Runs'.format(metric_title_name, dataset_title, method))
        plt.legend()

        # Show the plot
        plt.grid()
        plt.show()                     
                                    
def report_dropout_best_runs(metric_column, metric_name, metric_title_name):
    dropout_df = pd.read_csv('./experiment1/dropout/perf_dropout_results.csv')
    distinct_datasets = dropout_df['dataset'].unique()
    distinct_dropout = dropout_df['dropout'].unique()

    for dataset in distinct_datasets:
        filtered_dropout_df = dropout_df[(dropout_df['dataset'] == dataset)]
        distinct_methods = dropout_df['method'].unique()
        for method in distinct_methods:
            filtered_dropout_method_df = filtered_dropout_df[(dropout_df['method'] == method)]
            distinct_dropout = filtered_dropout_method_df['dropout'].unique()
            df_worst_validation_runs = None
            df_best_validation_runs = None
            dropout_dict = {}
            for dropout in distinct_dropout:
                filtered_dropout_df = filtered_dropout_method_df[(filtered_dropout_method_df['dropout'] == dropout)].sort_values(by=metric_column, ascending=True)  
                distinct_run = filtered_dropout_df['run'].unique()
                distinct_run.sort()  
                for run in distinct_run:
                    ind = run - 1
                    if ind in dropout_dict:
                        dropout_dict[ind] = pd.concat([dropout_dict[ind], filtered_dropout_df.iloc[[ind]]])
                    else:
                        dropout_dict[ind] = filtered_dropout_df.iloc[[ind]]

            for run in distinct_run:
                ind = run - 1
                df_run = dropout_dict[ind].sort_values(by='dropout', ascending=True).reset_index()
                df_run['dropout'] =  1-df_run['dropout']
                best_run = dropout_dict[distinct_run.max()-1].sort_values(by='dropout', ascending=True).reset_index()
                df_run['metric diff from best'] =  (df_run[metric_column] - best_run[metric_column]).round(4)
                print(df_run[['dataset', 'dropout', 'run', 'valid acc', 'valid mcc', 'metric diff from best']])
                y = df_run[metric_column]
                x = df_run['dropout']


                # Create a scatter plot
                plt.scatter(x, y, color='#003366')

                # Connect the points with a line
                if ind == distinct_run[-1] - 1:
                    plt.plot(x, y, color='#50C878', linestyle='-', linewidth=1, label='Best Runs')
                elif ind == 0:
                    plt.plot(x, y, color='#DC143C', linestyle='-', linewidth=1, label='Worst Runs')
                else:
                    plt.plot(x, y, color='#003366', linestyle='-', linewidth=1)

                # for i, txt in enumerate(df_run['run']):
                #     plt.annotate(txt, (x.iloc[[i]] + (0.01), y.iloc[[i]]+(y.iloc[[i]] * 0.0)))

            avg_metric_value = filtered_dropout_df[metric_column].mean()
            min_metric_value = filtered_dropout_df[metric_column].min()
            max_metric_value = filtered_dropout_df[metric_column].max()

            #plt.ylim(min_metric_value + (avg_metric_value * 0.5), max_metric_value - (avg_metric_value * 0.45))  # Set y-axis limits
 
            if dataset == 'stocknet':
                dataset_title = 'Acl18'
            else:
                dataset_title = dataset.title()
            # Adding labels and title
            plt.ylabel(metric_name)
            plt.xlabel('Dropout')
            plt.title('{0} vs Dropout for {1} Runs'.format(metric_title_name, dataset_title, method))
            plt.legend()

            # Show the plot
            plt.grid()
            plt.show()