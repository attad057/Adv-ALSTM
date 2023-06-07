import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, mean_squared_error, log_loss


def evaluate(prediction, ground_truth, hinge=False, reg=False, additional_metrics=False):
    assert ground_truth.shape == prediction.shape, 'shape mis-match'
    performance = {}
    if reg:
        performance['mse'] = mean_squared_error(np.squeeze(ground_truth), np.squeeze(prediction))
        return performance

    pred = label(hinge, prediction)
    try:
        performance['acc'] = accuracy_score(ground_truth, pred)
    except Exception:
        np.savetxt('prediction', pred, delimiter=',')
        exit(0)
    performance['mcc'] = matthews_corrcoef(ground_truth, pred)
    performance['pred'] = pred
    if additional_metrics == True:
        performance['ll'] = log_loss(ground_truth, prediction, eps = 1e-7)
    
    return performance

def label(hinge, prediction):
    if hinge:
        pred = (np.sign(prediction) + 1) / 2
        for ind, p in enumerate(pred):
            v = p[0]
            if abs(p[0] - 0.5) < 1e-8 or np.isnan(p[0]):
                pred[ind][0] = 0
    else:
        pred = np.round(prediction)

    return pred

def compare(current_performance, origin_performance):
    is_better = {}
    for metric_name in origin_performance.keys():
        if metric_name == 'mse':
            if current_performance[metric_name] < \
                    origin_performance[metric_name]:
                is_better[metric_name] = True
            else:
                is_better[metric_name] = False
        else:
            if current_performance[metric_name] > \
                    origin_performance[metric_name]:
                is_better[metric_name] = True
            else:
                is_better[metric_name] = False
    return is_better
