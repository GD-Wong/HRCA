import numpy as np
import pandas as pd
import scanpy as sc
import torch
import annotation
import model
import os
import sys
import timeit
import datetime
import gc
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--in_file', type=str, default='example/', help="path of input file")
parser.add_argument('--out_file', type=str, default='result/', help='path of save result')
parser.add_argument('--gpu_id', type=str, default='0', help='index of gpu to use, set "cpu" to use cpu')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--num_workers', type=int, default=4, help='number of cpu cores to use')
opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
file_path = opt.in_file
result_path = opt.out_file

ref_path = "Gene.h5ad"
os.makedirs(result_path, exist_ok=True)
enc = model.BERT(21333, head_number=8, n_layers=8)
clas = model.SENet(input_shape=1024, output_shape=36)
l_dictionary = np.load("dictionary.npy")
if opt.gpu_id == "cpu":
    enc.load_state_dict(torch.load("AE_200.pth", map_location=torch.device("cpu")))
    clas.load_state_dict(torch.load("predictor_200.pth", map_location=torch.device("cpu")))
else:
    enc.load_state_dict(torch.load("AE_200.pth"))
    clas.load_state_dict(torch.load("predictor_200.pth"))

if torch.cuda.is_available() and opt.gpu_id!="cpu":
    encoder = enc.cuda()
    classifier = clas.cuda()

for f in os.listdir(file_path):
    start = timeit.default_timer()
    sys.stdout.write("annotation for " + f + "\n")
    sys.stdout.flush()
    st = sc.read(file_path+f)
    anno_result = annotation.annotate(adata_file=file_path+f, reference_file=ref_path, encoder = enc, classifier = clas, l_dict = l_dictionary, result_name = "predicted", bs = opt.batch_size, workers = opt.num_workers)
    st.obs = pd.concat([st.obs, anno_result], axis = 1)
    st.write(result_path+f)
    sys.stdout.write("end annotation \n")
    sys.stdout.write("ETA: %s  \n" % datetime.timedelta(seconds=timeit.default_timer() - start))
    sys.stdout.flush()
    gc.collect()

