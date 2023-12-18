from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss, mean_squared_error
import numpy as np

def calc_auc_score(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


def calc_f1_score_macro(y_true, y_pred):
    """
    y_pred: softmax後の値
    """
    y_pred = y_pred.argmax(1)
    return f1_score(y_true, y_pred, average='macro')

def calc_acc_score_binary(y, prediction):
    prediction = (prediction >= 0.5).astype(int)
    return accuracy_score(y, prediction)

def calc_acc_score_multiclass(y_true, y_pred):
    """
    y_pred: softmax後の値
    """
    y_pred = y_pred.argmax(1)
    return accuracy_score(y_true, y_pred)


def calc_binary_f1(y, prediction):
    prediction = (prediction >= 0.5).astype(int)
    return f1_score(y, prediction)


def calc_binary_f1_search_thres(y, prediction):
    # thres = [0.35, 0.36, 0.37, 0.38, 0.39, 0.40, 0.41, 0.42, 0.43, 0.44]
    thres = np.array(range(25, 40)) / 100

    best_score = -1
    best_th = 0
    for th in thres:
        print(f'th: {th}')
        prediction_binary = (prediction > th).astype(int)
        score = f1_score(y, prediction_binary)
        print(f'score: {score}')
        if score > best_score:
            best_score = score
            best_th = th
    print(f'best_th: {best_th}')
    print(f'best_score: {best_score}')

    return best_score

def calc_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def MCRMSE(y_trues, y_preds):
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:, i]
        y_pred = y_preds[:, i]
        score = mean_squared_error(y_true, y_pred, squared=False)  # RMSE
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores


"""
def get_score(y_true, y_pred):
    return acc_score(y_true, y_pred)
    # return f1_score_macro(y_true, y_pred)
    # return calc_binary_f1(y_true, y_pred)
    # return acc_score(y_true, y_pred)
    # return f1_score(y_true, y_pred)
    # return calc_rmse(y_true, y_pred)
    # return MCRMSE(y_true, y_pred)
"""
