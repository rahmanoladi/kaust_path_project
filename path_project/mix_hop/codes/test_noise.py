


import shutil
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import dgl, dgllife
import time
from functools import partial

from torch.utils.data import DataLoader

from dgllife.data import FreeSolv
from dgllife.utils import EarlyStopping, smiles_to_bigraph, CanonicalAtomFeaturizer, AttentiveFPAtomFeaturizer, CanonicalBondFeaturizer, AttentiveFPBondFeaturizer, ConsecutiveSplitter, ScaffoldSplitter, RandomSplitter
from dgllife.utils import Meter, SMILESToBigraph

from mix_hop_model import mix_hop_model


import os
import re
import sys

def get_utils_path():
    path = os.getcwd()
    dirs = re.split('/', path)
    if dirs[0] == '':
        root_dirs = dirs[1:-2]
    else:
        root_dirs =   dirs[0:-2]

    root_path = '/'

    for dir in root_dirs:
        root_path = os.path.join(root_path, dir)

    utils_path = os.path.join(root_path, 'utils')
    return utils_path
    

utils_path = get_utils_path()
print("UTILS_PATH: ", utils_path)
sys.path.insert(1, utils_path)



from utils import normalize_adj, max_nodes_and_edges, max_path_len, early_stopping, beta_mat_3d_from_dgl_graph, split_mat_3d, pad_mat_3d, split_mat_2d, pad_mat_2d, split_pad_and_stack_beta, split_pad_and_stack_adj, split_pad_and_stack_feat
from utils import aws_collate_molgraphs as collate_molgraphs





def float_or_str(value):
    try:
        return float(value)
    except:
        return value


def get_feat_dim(data_loader):
    for i, batch in enumerate(data_loader):
        smiles, dgl_graph, labels, masks = batch
        feats = dgl_graph.ndata['h']
        inp_dim = feats.shape[-1]
        edge_dim = dgl_graph.edata['e'].shape[-1]
        return inp_dim, edge_dim          




def add_noise_0(dataset, factor):
    
    data_loader = DataLoader(dataset= dataset, shuffle=False, batch_size=len(dataset), collate_fn=collate_molgraphs, num_workers=args.num_workers)

    for i, batch in enumerate(data_loader):
        
        smiles, dgl_graph, labels, masks = batch
        node_feats = dgl_graph.ndata['h'] 
        edge_feats = dgl_graph.edata['e']

        node_mean = torch.mean(node_feats, dim = 0)
        node_std = torch.std(node_feats, dim = 0)
        node_std = node_std*factor

        edge_mean = torch.mean(edge_feats, dim = 0)
        edge_std = torch.std(edge_feats, dim = 0)
        edge_std = edge_std*factor
    
    
    for j in range(len(dataset)):
        smiles, dgl_graph, labels = dataset[j]
        node_feats = dgl_graph.ndata['h']
        node_z = torch.randn(node_feats.shape[0], node_feats.shape[1])
        node_noise = torch.mul(node_z, node_std) + node_mean
        dgl_graph.ndata['h'] = node_feats + node_noise        

        edge_feats = dgl_graph.edata['e']
        edge_z = torch.randn(edge_feats.shape[0], edge_feats.shape[1])
        edge_noise = torch.mul(edge_z, edge_std) + edge_mean
        dgl_graph.edata['e'] = edge_feats + edge_noise 
  
        dataset[j] =  smiles, dgl_graph, labels  

    return dataset   
 



def node_mean_and_std(dataset):
    
    data_loader = DataLoader(dataset= dataset, shuffle=False, batch_size=len(dataset), collate_fn=collate_molgraphs, num_workers=args.num_workers)

    for i, batch in enumerate(data_loader):
        
        smiles, dgl_graph, labels, masks = batch
        node_feats = dgl_graph.ndata['h'] 
        edge_feats = dgl_graph.edata['e']

        node_mean = torch.mean(node_feats, dim = 0)
        node_std = torch.std(node_feats, dim = 0)
        
    return node_mean, node_std 
        
def get_beta_max(beta_list):
    len_beta = len(beta_list)
    max_val = 0
    for i in range(len_beta):
        val = np.amax(beta_list[i])
        if val > max_val:
            max_val = val
    return max_val


def get_max_path_len(data_loader):
    max_path_len = 0
    for i, batch in enumerate(data_loader):
        smiles, dgl_graph, labels, masks = batch
        dist_mat, path = dgl.shortest_dist(dgl_graph, return_paths=True)
        
        if path.shape[2] > max_path_len:
            max_path_len = path.shape[2]
 
    return max_path_len  


def add_noise_to_node(dgl_graph, seed):
        torch.manual_seed(seed)
        node_feats = dgl_graph.ndata['h']
        node_z = torch.randn(node_feats.shape[0], node_feats.shape[1])
        node_noise = torch.mul(node_z, args.node_std) + args.node_mean
        dgl_graph.ndata['h'] = node_feats + node_noise 
        return dgl_graph





def train(data_loader, device, max_nodes, model, loss_function, optimizer):
    
    loss_list = []
    model.train()
    train_meter = Meter()
    cum_loss = 0.
    num_graphs = 0
    for i, batch in enumerate(data_loader):
 
        smiles, dgl_graph, labels, masks = batch 

        dgl_graph = add_noise_to_node(dgl_graph, i) 


        normalized_adj = normalize_adj(dgl_graph, max_nodes)        
        labels, masks, normalized_adj = labels.to(device), masks.to(device), normalized_adj.to(device)

        numb_nodes = dgl_graph.num_nodes()
        batch_num_nodes = dgl_graph.batch_num_nodes()
        feats = dgl_graph.ndata['h'] 

       

        feats = feats.numpy()
        feats = split_pad_and_stack_feat(feats, batch_num_nodes, max_nodes)
        feats = torch.from_numpy(feats)
        feats = feats.to(torch.float32).to(device)   
        out = model(feats, normalized_adj)
           
        loss = (loss_function(out, labels) * (masks != 0).float()).mean() 
        
          
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       
        train_meter.update(out, labels, masks)
    train_score = np.mean(train_meter.compute_metric(args.metric))
    return loss, train_score




@torch.no_grad()
def test(data_loader, device, max_nodes, model, loss_function):
    
    loss_list = []
    model.eval()
    train_meter = Meter()
    cum_loss = 0.
    num_graphs = 0
    for i, batch in enumerate(data_loader):
 
        smiles, dgl_graph, labels, masks = batch 
        dgl_graph = add_noise_to_node(dgl_graph, i) 
        normalized_adj = normalize_adj(dgl_graph, max_nodes)
        labels, masks, normalized_adj = labels.to(device), masks.to(device), normalized_adj.to(device)


        numb_nodes = dgl_graph.num_nodes()
        batch_num_nodes = dgl_graph.batch_num_nodes()
        feats = dgl_graph.ndata['h'] 
        feats = feats.numpy()
        feats = split_pad_and_stack_feat(feats, batch_num_nodes, max_nodes)
        feats = torch.from_numpy(feats)
        feats = feats.to(torch.float32).to(device)   
        out = model(feats, normalized_adj)
        
           
        loss = (loss_function(out, labels) * (masks != 0).float()).mean() 
               
        train_meter.update(out, labels, masks)
    train_score = np.mean(train_meter.compute_metric(args.metric))
    return loss, train_score

 
 
def main():

    device = torch.device("cuda" if args.use_gpu == 1 and torch.cuda.is_available() else 'cpu')  
    args.node_mean, args.node_std = node_mean_and_std(dataset, collate_molgraphs, args.num_workers)
    if args.splitter == 'consec':
        splitter = ConsecutiveSplitter()
        split = splitter.train_val_test_split(dataset)
    elif args.splitter == 'random':
        splitter = RandomSplitter()
        split = splitter.train_val_test_split(dataset, random_state = args.rand_state)
    elif args.splitter == 'scaffold':
        splitter = ScaffoldSplitter()
        split = splitter.train_val_test_split(dataset,  scaffold_func='smiles')
   
# Deliberately not shuffling training set
    train_loader = DataLoader(dataset=split[0], shuffle=False, batch_size=args.batch_size, collate_fn=collate_molgraphs, num_workers=args.num_workers)
    val_loader = DataLoader(dataset=split[1], shuffle = False, batch_size=args.batch_size, collate_fn=collate_molgraphs, num_workers=args.num_workers)
    test_loader = DataLoader(dataset=split[2], shuffle = False, batch_size=args.batch_size, collate_fn=collate_molgraphs, num_workers=args.num_workers)



    # load model
    inp_dim, edge_dim = get_feat_dim(train_loader) 
    #model = graphormer_model(args.use_path_info, args.num_layers, inp_dim, edge_dim, args.hidden_dim, args.num_heads, args.ff_dim, args.num_classes, args.max_len)
    if args.use_path_info == 0:
        args.max_pow = 1
    model = mix_hop_model(args.num_layers, inp_dim, args.small_hidden_dim, args.num_classes, args.max_pow, args.dropout)
    model = model.to(device = device)
    
    model_name =  'model' + '_dat_' + args.dataset + '_path_' + str(args.use_path_info) + '_rep_' + str(args.repitition) + '.pth'
    stopper = EarlyStopping(patience= args.early_patience, filename=os.path.join(args.dir_to_save_model, model_name), metric=args.metric)

  
    if (args.opt).lower() == 'adam':
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)
    elif (args.opt).lower() == 'sgd':
        opt = optim.SGD(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)

    if args.metric in [ 'r2', 'mae', 'rmse']:
        loss_fn = nn.SmoothL1Loss(reduction='none')       
    elif args.metric in ['roc_auc_score', 'pr_auc_score'] :
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    
    for epoch in range(args.epochs):
        # Train
        if epoch == 0:
            t1 = time.time()

        #(data_loader, device, max_nodes, max_path_len, model, loss_function, optimizer)
        train_loss, train_score = train(train_loader, device, args.max_nodes, model, loss_fn, opt) 
        if epoch == 0:
            t2 = time.time()
            print("Time spent on single epoch of train function in seconds: ", (t2- t1))
            print('---------- Training ----------')
        print('| epoch {:3d} | train loss {:5.5f} | train {} {:2.3f}'.format(epoch + 1, train_loss,  args.metric, train_score,), flush=True)
      
        val_loss, val_score =   test(val_loader, device, args.max_nodes, model, loss_fn)
        early_stop = stopper.step(val_score, model)
        print('epoch {:d}/{:d}, validation {} {:.4f}, best validation {} {:.4f}'.format(epoch + 1, args.epochs, args.metric, val_score, args.metric, stopper.best_score), flush=True)

        if early_stop:
            break 

    stopper.load_checkpoint(model)
    val_loss, val_score =   test(val_loader, device, args.max_nodes,  model, loss_fn)
    test_loss, test_score = test(test_loader, device, args.max_nodes,  model, loss_fn) 
    print('Final validation loss {:5.5f} | Final validation {} {:2.3f}'.format( val_loss, args.metric, val_score,), flush=True)
    print('Final test loss {:5.5f} | Final test {} {:2.3f}'.format( test_loss, args.metric, test_score,), flush=True)

        

    #feats = test_feats(test_loader, device , model, args.max_nodes, args.max_path_len)
    
if __name__ == '__main__':
    from argparse import ArgumentParser

    #from utils import init_featurizer, mkdir_p, split_dataset, get_configure

    parser = ArgumentParser('Moleculenet')
    parser.add_argument('-d', '--dataset', type = str, help='Dataset to use')
    parser.add_argument('--use_path_info', type=int, default=None, help='Max nodes per graph in dataset')
    parser.add_argument('--repitition', type=int, default=None, help='Max nodes per graph in dataset')
    parser.add_argument('--max_nodes', type=int, default=None, help='Max nodes per graph in dataset')
    parser.add_argument('--num_classes', type=int, default= None, help='Number of classes')  
    parser.add_argument('--splitter', type=str, default=None, help='Data splitting method. Choose from: [scaffold, random, consec ]') 
    parser.add_argument('--rand_state', type=float, default=None, help='random seed when using random splitting method for dataset')  
    parser.add_argument('-af', '--atom_featurizer', type=str, default = None, help='Featurization for atoms')              
    #parser.add_argument('-bf', '--bond_featurizer', type = str, default=None, help='Featurization for bonds')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train.')
    parser.add_argument('--num-layers', type=int, default=None, help='Number of layers.')
    parser.add_argument('--small_hidden_dim', type=int, default= None, help='hidden layer(s) dimension')
    parser.add_argument('--max_pow', type=int, default= None, help='hidden layer(s) dimension')
    parser.add_argument('--dropout', type=float, default=None, help='dropout')
    parser.add_argument('--weight_decay', type=float, default=None, help='weight decay')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size.')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate.')

    parser.add_argument('-me', '--metric', choices=['roc_auc_score', 'pr_auc_score', 'r2', 'mae', 'rmse'], default=None, help='Metric for evaluation (default: roc_auc_score)')
    parser.add_argument('-nw', '--num-workers', type=int, default=0, help='Number of processes for data loading (default: 0)')    
    parser.add_argument('--opt', type=str, default= 'adam', help='Optimizer')   
    parser.add_argument('--early_stopping', type=int, default=0, help='Whether or not to use early stopping.')
    parser.add_argument('--early_patience', type=int, default=0, help='Number of epochs to wait before stopping training')
    parser.add_argument('--early_delta', type=float, default=0, help='Min value that difference between new metric and old metric must exceed to increment early stopping counter')     
    parser.add_argument('--edge_norm', type=str, default='both', help='edges weights normalization type')      
    parser.add_argument('--use_gpu', type=int, default= None, help='transductive or inductive setting?')
    parser.add_argument('--max_len', type=int, default= None, help='power dimension + 1')
    parser.add_argument('--dir_to_save_model', type=str, default= '/ibex/user/ibraheao/path_project/mix_hop/model_directory', help='power dimension + 1')

    
    args = parser.parse_args()
        

    
    print("dgl version: ", dgl.__version__)
    print("dgllife version: ", dgllife.__version__)

    if args.atom_featurizer == 'canonical':
        node_featurizer=CanonicalAtomFeaturizer()
        edge_featurizer = CanonicalBondFeaturizer()
        
        smiles_to_g = SMILESToBigraph(add_self_loop=False, node_featurizer= node_featurizer,
                                  edge_featurizer=edge_featurizer)

    elif args.atom_featurizer == 'attentivefp':
        node_featurizer=AttentiveFPAtomFeaturizer()
        edge_featurizer = AttentiveFPBondFeaturizer()
        smiles_to_g = SMILESToBigraph(add_self_loop=False, node_featurizer= node_featurizer,
                                  edge_featurizer=edge_featurizer)


    if args.dataset == 'MUV':
        from dgllife.data import MUV
        dataset = MUV(smiles_to_graph=smiles_to_g,
                      n_jobs=1 if args.num_workers == 0 else args.num_workers)
         
    elif args.dataset == 'BACE':
        from dgllife.data import BACE
        dataset = BACE(smiles_to_graph=smiles_to_g,
                      n_jobs=1 if args.num_workers == 0 else args.num_workers)
        args.metric = 'roc_auc_score'
    elif args.dataset == 'BBBP':
        from dgllife.data import BBBP
        dataset = BBBP(smiles_to_graph=smiles_to_g,
                       n_jobs=1 if args.num_workers == 0 else args.num_workers)
        args.metric = 'roc_auc_score'
    elif args.dataset == 'ClinTox':
        from dgllife.data import ClinTox
        dataset = ClinTox(smiles_to_graph=smiles_to_g,
                          n_jobs=1 if args.num_workers == 0 else args.num_workers)
        args.metric = 'roc_auc_score'
    elif args.dataset == 'SIDER':
        from dgllife.data import SIDER
        dataset = SIDER(smiles_to_graph=smiles_to_g,
                        n_jobs=1 if args.num_workers == 0 else args.num_workers)
        args.metric = 'roc_auc_score'
    elif args.dataset == 'ToxCast':
        from dgllife.data import ToxCast
        dataset = ToxCast(smiles_to_graph=smiles_to_g,
                          n_jobs=1 if args.num_workers == 0 else args.num_workers)
        args.metric = 'roc_auc_score'
    elif args.dataset == 'HIV':
        from dgllife.data import HIV
        dataset = HIV(smiles_to_graph=smiles_to_g,
                      n_jobs=1 if args.num_workers == 0 else args.num_workers)
        args.metric = 'roc_auc_score'
    elif args.dataset == 'PCBA':
        from dgllife.data import PCBA
        dataset = PCBA(smiles_to_graph=smiles_to_g,
                       n_jobs=1 if args.num_workers == 0 else args.num_workers)
    elif args.dataset == 'Tox21':
        from dgllife.data import Tox21
        dataset = Tox21(smiles_to_graph=smiles_to_g,
                        n_jobs=1 if args.num_workers == 0 else args.num_workers)
        args.metric = 'roc_auc_score'
    elif args.dataset == 'FreeSolv':
        from dgllife.data import FreeSolv
        dataset = FreeSolv(smiles_to_graph=smiles_to_g,
                        n_jobs=1 if args.num_workers == 0 else args.num_workers)
        args.metric = 'rmse'

    elif args.dataset == 'ESOL':
        from dgllife.data import ESOL
        dataset = ESOL(smiles_to_graph=smiles_to_g,
                        n_jobs=1 if args.num_workers == 0 else args.num_workers)
        args.metric = 'rmse'

    elif args.dataset == 'Lipophilicity' :
        from dgllife.data import Lipophilicity 
        dataset = Lipophilicity(smiles_to_graph=smiles_to_g,
                        n_jobs=1 if args.num_workers == 0 else args.num_workers)
        args.metric = 'rmse'
    else:
        raise ValueError('Unexpected dataset: {}'.format(args['dataset']))

    args.n_tasks = dataset.n_tasks
    args.num_classes = args.n_tasks
    args.max_nodes, _ = max_nodes_and_edges(dataset)

    
   

    print(args)
      
    main()