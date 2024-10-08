import torch
import scanpy as sc
import pandas as pd
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from scipy import sparse

# 输入空间转录组数据准备转换
class InputLoader(Dataset):
    def __init__(self, input_file, ref_file, norm_flag = False, dropout_flag = False, dropout_rate=0.7):
        input_adata = sc.read(input_file)
        ref_adata = sc.read(ref_file)
        genes = list(ref_adata.var.index)
        all_adata = sc.concat({"input":input_adata, "ref":ref_adata}, label = "data_type", join="outer", axis=0)[:, genes]
        input_adata = all_adata[all_adata.obs.data_type=="input", :]
        sc.pp.filter_cells(input_adata, min_genes=80)
        if norm_flag:
            sc.pp.normalize_total(input_adata, target_sum=1e4)
            sc.pp.log1p(input_adata)
        self.input = input_adata.X
        self.obs = input_adata.obs
        self.var = input_adata.var
        self.obsm = input_adata.obsm
        self.dropout_rate = dropout_rate
        self.dropout_flag = dropout_flag
    def __len__(self):
        return self.input.shape[0]
    def __getitem__(self, idx):
        spot = self.input[idx, :]
        if sparse.issparse(spot):
            spot = spot.toarray()
        spot = torch.squeeze(torch.FloatTensor(spot))
        spot = dropout(spot, dropout_rate=self.dropout_rate, dropout_flag=self.dropout_flag)
        return spot

def normalize_torch(x, scalefactor = 10000, log_flag = True):
    sum_counts = x.sum(axis=1)
    x = (x*scalefactor).div(sum_counts.unsqueeze(1))
    if log_flag:
        x = x.log1p()
    return x

def dropout(x, dropout_rate = 0.7, dropout_flag = True, norm_flag = True):
    mask = torch.rand(*x.shape, device=x.device)>dropout_rate
    if dropout_rate == 0:
        return x
    if dropout_flag:
        if norm_flag:
            return normalize_torch(x*mask, scalefactor=10000, log_flag=True)
        else:
            return x*mask
    else:
        return x

# 将label数字转为字符,label_int为列表
def label_int_to_str(label_int, dict_idx):
    labels = pd.DataFrame(label_int)
    for i in range(len(dict_idx)):
        labels = labels.replace(i, dict_idx[i])
    labels = list(labels.iloc[:, 0])
    return labels

def annotate(adata_file, reference_file, encoder, classifier, l_dict, result_name = "predicted", bs = 128, workers = 4):
    encoder.eval()
    classifier.eval()
    input_dataloader = DataLoader(
        InputLoader(input_file=adata_file, ref_file=reference_file, norm_flag=True, dropout_flag=False),
        batch_size=bs,
        shuffle=False,
        num_workers=workers
    )
    all_pred = torch.tensor([], dtype=int)
    for i, source in enumerate(input_dataloader):
        if torch.cuda.is_available():
            source_input = Variable(source).cuda()
        else:
            source_input = Variable(source)
        probs = classifier(encoder.encode(source_input))
        pred = probs.argmax(dim=1)
        pred_cpu = pred.cpu()
        all_pred = torch.hstack((all_pred, pred_cpu))
    pred_result = pd.DataFrame(label_int_to_str(all_pred, l_dict), columns=[result_name], index=input_dataloader.dataset.obs.index)
    return pred_result
