import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


DRA_valid_results = pd.read_csv('NFM-DRA/DRA/experiment/result_valid.txt', sep='\s+', header=None)


DRA_valid_prob = np.array(DRA_valid_results[1].to_list())


DRA_JSIEC_results = pd.read_csv('NFM-DRA/DRA/experiment/result_JSIEC.txt', sep='\s+', header=None)


DRA_JSIEC_prob = np.array(DRA_JSIEC_results[1].to_list())


DRA_RIADD_results = pd.read_csv('NFM-DRA/DRA/experiment/result_RIADD.txt', sep='\s+', header=None)


DRA_RIADD_prob = np.array(DRA_RIADD_results[1].to_list())


NFM_JSIEC_results = pd.read_csv('NFM-DRA/NFM/results/Fundus_Results_224_224_0.1/JSIEC_results.txt', sep='\s+', header=None)


NFM_JSIEC_prob = np.array(NFM_JSIEC_results[1].to_list())


NFM_RIADD_results = pd.read_csv('NFM-DRA/NFM/results/Fundus_Results_224_224_0.1/RIADD_results.txt', sep='\s+', header=None)


NFM_RIADD_prob = np.array(NFM_RIADD_results[1].to_list())


def rescale_DRA_by_validset(DRA_valid_prob, DRA_JSIEC_prob, DRA_RIADD_prob):
    rescaled_DRA_JSIEC_prob = (DRA_JSIEC_prob - DRA_valid_prob.min()) / (DRA_valid_prob.max() - DRA_valid_prob.min())
    rescaled_DRA_RIADD_prob = (DRA_RIADD_prob - DRA_valid_prob.min()) / (DRA_valid_prob.max() - DRA_valid_prob.min())
    return rescaled_DRA_JSIEC_prob, rescaled_DRA_RIADD_prob


rescaled_DRA_JSIEC_prob, rescaled_DRA_RIADD_prob = rescale_DRA_by_validset(DRA_valid_prob, DRA_JSIEC_prob, DRA_RIADD_prob)


final_JSIEC_prob = 0.5 * (rescaled_DRA_JSIEC_prob + NFM_JSIEC_prob)
final_RIADD_prob = 0.5 * (rescaled_DRA_RIADD_prob + NFM_RIADD_prob)


JSIEC_gt = DRA_JSIEC_results[0].to_list()
RIADD_gt = DRA_RIADD_results[0].to_list()


# cal auroc
JSIEC_auc = roc_auc_score(JSIEC_gt, final_JSIEC_prob)
RIADD_auc = roc_auc_score(RIADD_gt, final_RIADD_prob)

print(f"JSIEC_auc: {JSIEC_auc}, RIADD_auc: {RIADD_auc}")


