import numpy as np
import random
def Error_check(settings):
    classes_num = torch.tensor(settings['number_of_classes'])

    
    if settings['data_set'] == 'synthetic':
        if len(settings['p'])!=classes_num.sum():
            raise ValueError('p should have {} elements'.format(classes_num))
        
        if len(settings['mu'])!=classes_num.sum():
            raise ValueError('mu should have {} elements'.format(classes_num))
        if settings['num_runs']<settings['num_threads']:
            raise ValueError('num_runs should be >= num_threads')
    
    
import yaml

# Specify the file path from which you want to read the YAML file

import torch       
def make_settings(args,wandb_conf):
            
    # if args.data_set != 'synthetic':
    
    from Graph.data_config import data_set_config
    # with open('./Graph/data_config.py', 'r') as yaml_file:
    #     data_set_config = yaml.safe_load(yaml_file)[]
    data_set_config = data_set_config[wandb_conf.get('data_set',args.data_set)]
    if args.classes_to_keep is not None:
        data_set_config['number_of_classes'] = len(args.classes_to_keep)
    # from model.model_plans import model_config

    # settings['imbalance_map']
    #too cleanup
    # edge_temper_presets = wandb_conf.get('edge_temper_presets',args.edge_temper_presets)
    # edge_dist=torch.tensor(data_set_config['edge_temper_presets'][edge_temper_presets])*args.q
    # raise ValueError(args.p)
    # p = wandb_conf.get('p',args.p)
    # q = wandb_conf.get('q',args.q)
    # for i in range(len(edge_dist)):

    #     edge_dist[i,i]*=(p/(q+1e-7))
    imbalance_classes = [1 if i<(data_set_config['number_of_classes']-data_set_config['number_of_classes']//2) else 1/args.imbal_coef for i in range(data_set_config['number_of_classes'])]
    # raise ValueError(imbalance_classes)
    
    settings={
        'number_of_classes':data_set_config['number_of_classes'],
        # 'dist':data_set_config['dist'],
        # 'p':torch.tensor(args.p),
        # 'q':torch.tensor(args.q),
        'mu':torch.tensor([[mu_value for _ in range(args.features_num)] for mu_value in args.mu]),
        'num_runs':args.num_runs,
        'wandb_profile':args.wandb_profile,
        'project_name':args.project_name,
        'run_name':args.run_name,
        'loss_type':args.loss_type,
        'tea':args.titled_tea,
        'focal_gamma':args.focal_gamma,
        'aggr_methode':args.aggr_methode,
        'lr':args.lr,
        'wd':args.wd,
        'hidden_channels':args.hidden_channels,
        'num_heads':args.num_heads,
        'weight_decay':args.wd,
        'use_scheduler':args.use_scheduler,
        'schedule_factor':args.schedule_factor,
        'schedule_patience':args.schedule_patience,
        'regularizer':args.regularizer,
        'regularizer_coef':args.regularizer_coef,
        'kl_power':args.kl_power,
        'model_type':args.model_type,
        'data_set':args.data_set,
        'data_set_conf':data_set_config['data'],
        'data_path':data_set_config['data_path'],
        'pca_dim':args.pca_dim,
        'lda_dim':args.lda_dim,
        'debug':args.debug,
        'imbalance_map':imbalance_classes,
        'num_extra_runs':args.num_extra_runs,
        'pseudo_thr_band':args.pseudo_thr_band,
        'pseudo_labels_thr':args.pseudo_labels_thr,
        'ens':args.ens,
        'pred_temp':args.pred_temp,
        'keep_prob':args.keep_prob,
        'device':args.device,
        # 'predictor':args.predictor,
        # 'generator':args.generator,
        # 'top_percent_link':args.top_percent_link/100,
        
        ##TAM##
        'tam':args.tam,
        'warm_up':args.warm_up,
        'tam_alpha':args.tam_alpha,
        'tam_beta':args.tam_beta,
        'temp_phi':args.temp_phi,
        'pseudo_labeler':args.pseudo_labeler,
        'mapping':args.mapping,
        
        'entropy_uncertainty_percent':args.entropy_uncertainty_percent,
        'entropy_uncertainty':args.entropy_uncertainty,
        'uncertainty':args.uncertainty,
        'change_order':args.change_order,
        'classes_to_keep':args.classes_to_keep,
        'num_augmentations':args.num_augmentations,
        'nums_to_remove':args.nums_to_remove
        }
    settings.update(wandb_conf)
    return settings


def set_seed(seed):
       
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)






import torch
import numpy as np
from torch_scatter import scatter_add

def get_dataset(name, path, split_type='public'):
    import torch_geometric.transforms as T 

    transform = T.Compose([T.NormalizeFeatures(), T.ToUndirected()])
    if name == "Cora" or name == "CiteSeer" or name == "PubMed":
        from torch_geometric.datasets import Planetoid

        dataset = Planetoid(path, name, transform=transform, split=split_type)
        
    elif name == "chameleon" or name == "squirrel":
        from dataset.WikipediaNetwork import WikipediaNetwork
        dataset = WikipediaNetwork(path, name, transform = transform)


    elif name == "Wisconsin":
        # from torch_geometric.datasets import  WebKB
        from dataset.WebKB import WebKB
        dataset = WebKB(path, name, transform = transform)
    else:
        raise NotImplementedError("Not Implemented Dataset!")

    return dataset





def get_idx_info(label, n_cls, train_mask):
    
    index_list = torch.arange(len(label)).to('cuda:0')
    idx_info = []
    for i in range(n_cls):
        # raise ValueError(((label == i)& train_mask).'cuda:0')
        cls_indices = index_list[((label == i) & train_mask)]
        idx_info.append(cls_indices)
    return idx_info



import argparse
import scipy.sparse as sp
import numpy as np
import torch
import ipdb
from scipy.io import loadmat
import networkx as nx
import multiprocessing as mp
import torch.nn.functional as F
from functools import partial
import random
from sklearn.metrics import roc_auc_score, f1_score
from copy import deepcopy
from scipy.spatial.distance import pdist,squareform

# def get_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--no-cuda', action='store_true', default=False,
#                     help='Disables CUDA training.')
#     parser.add_argument('--seed', type=int, default=42)
#     parser.add_argument('--nhid', type=int, default=64)
#     parser.add_argument('--dataset', type=str, default='cora')
#     parser.add_argument('--size', type=int, default=100)

#     parser.add_argument('--epochs', type=int, default=2010,
#                 help='Number of epochs to train.')
#     parser.add_argument('--lr', type=float, default=0.001)
#     parser.add_argument('--weight_decay', type=float, default=5e-4)
#     parser.add_argument('--dropout', type=float, default=0.1)

#     parser.add_argument('--batch_nums', type=int, default=6000, help='number of batches per epoch')
#     parser.add_argument('--batch_size', type=int, default=40, help='number of batches per epoch')


#     parser.add_argument('--imbalance', action='store_true', default=False)
#     parser.add_argument('--setting', type=str, default='no', 
#         choices=['no','upsampling', 'smote','reweight','embed_up', 'recon','newG_cls','recon_newG'])
#     #upsampling: oversample in the raw input; smote: ; reweight: reweight minority classes; 
#     # embed_up: 
#     # recon: pretrain; newG_cls: pretrained decoder; recon_newG: also finetune the decoder

#     parser.add_argument('--opt_new_G', action='store_true', default=False) # whether optimize the decoded graph based on classification result.
#     parser.add_argument('--load', type=str, default=None)
#     parser.add_argument('--up_scale', type=float, default=1)
#     parser.add_argument('--im_ratio', type=float, default=0.5)
#     parser.add_argument('--rec_weight', type=float, default=0.000001)
#     parser.add_argument('--model', type=str, default='sage', 
#         choices=['sage','gcn','GAT'])



#     return parser

def split_arti(labels, c_train_num):
    #labels: n-dim Longtensor, each element in [0,...,m-1].
    #cora: m=7
    num_classes = len(set(labels.tolist()))
    c_idxs = [] # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)
    c_num_mat[:,1] = 25
    c_num_mat[:,2] = 55

    for i in range(num_classes):
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        print('{:d}-th class sample number: {:d}'.format(i,len(c_idx)))
        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        train_idx = train_idx + c_idx[:c_train_num[i]]
        c_num_mat[i,0] = c_train_num[i]

        val_idx = val_idx + c_idx[c_train_num[i]:c_train_num[i]+25]
        test_idx = test_idx + c_idx[c_train_num[i]+25:c_train_num[i]+80]

    random.shuffle(train_idx)

    #ipdb.set_trace()

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)
    #c_num_mat = torch.LongTensor(c_num_mat)


    return train_idx, val_idx, test_idx, c_num_mat

def split_genuine(labels):
    #labels: n-dim Longtensor, each element in [0,...,m-1].
    #cora: m=7
    num_classes = len(set(labels.tolist()))
    c_idxs = [] # class-wise index
    train_idx = []
    val_idx = []
    test_idx = []
    c_num_mat = np.zeros((num_classes,3)).astype(int)

    for i in range(num_classes):
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        c_num = len(c_idx)
        print('{:d}-th class sample number: {:d}'.format(i,len(c_idx)))
        random.shuffle(c_idx)
        c_idxs.append(c_idx)

        if c_num <4:
            if c_num < 3:
                print("too small class type")
                ipdb.set_trace()
            c_num_mat[i,0] = 1
            c_num_mat[i,1] = 1
            c_num_mat[i,2] = 1
        else:
            c_num_mat[i,0] = int(c_num/4)
            c_num_mat[i,1] = int(c_num/4)
            c_num_mat[i,2] = int(c_num/2)


        train_idx = train_idx + c_idx[:c_num_mat[i,0]]

        val_idx = val_idx + c_idx[c_num_mat[i,0]:c_num_mat[i,0]+c_num_mat[i,1]]
        test_idx = test_idx + c_idx[c_num_mat[i,0]+c_num_mat[i,1]:c_num_mat[i,0]+c_num_mat[i,1]+c_num_mat[i,2]]

    random.shuffle(train_idx)

    #ipdb.set_trace()

    train_idx = torch.LongTensor(train_idx)
    val_idx = torch.LongTensor(val_idx)
    test_idx = torch.LongTensor(test_idx)
    #c_num_mat = torch.LongTensor(c_num_mat)


    return train_idx, val_idx, test_idx, c_num_mat


def print_edges_num(dense_adj, labels):
    c_num = labels.max().item()+1
    dense_adj = np.array(dense_adj)
    labels = np.array(labels)

    for i in range(c_num):
        for j in range(c_num):
            #ipdb.set_trace()
            row_ind = labels == i
            col_ind = labels == j

            edge_num = dense_adj[row_ind].transpose()[col_ind].sum()
            # print("edges between class {:d} and class {:d}: {:f}".format(i,j,edge_num))


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def print_class_acc(output, labels, class_num_list, pre='valid'):
    pre_num = 0
    #print class-wise performance
    '''
    for i in range(labels.max()+1):
        
        cur_tpr = accuracy(output[pre_num:pre_num+class_num_list[i]], labels[pre_num:pre_num+class_num_list[i]])
        print(str(pre)+" class {:d} True Positive Rate: {:.3f}".format(i,cur_tpr.item()))

        index_negative = labels != i
        labels_negative = labels.new(labels.shape).fill_(i)
        
        cur_fpr = accuracy(output[index_negative,:], labels_negative[index_negative])
        print(str(pre)+" class {:d} False Positive Rate: {:.3f}".format(i,cur_fpr.item()))

        pre_num = pre_num + class_num_list[i]
    '''

    #ipdb.set_trace()
    if labels.max() > 1:
        auc_score = roc_auc_score(labels.detach().cpu(), F.softmax(output.cpu(), dim=-1).detach(), average='macro', multi_class='ovr')
    else:
        auc_score = roc_auc_score(labels.detach().cpu(), F.softmax(output.cpu(), dim=-1)[:,1].detach(), average='macro')

    macro_F = f1_score(labels.detach().cpu(), torch.argmax(output.cpu(), dim=-1).detach(), average='macro')
    print(str(pre)+' current auc-roc score: {:f}, current macro_F score: {:f}'.format(auc_score,macro_F))

    return

def src_upsample(adj,features,labels,idx_train, portion=1.0, im_class_num=3):
    c_largest = labels.max().item()
    adj_back = adj.to_dense()
    chosen = None

    #ipdb.set_trace()
    avg_number = int(idx_train.shape[0]/(c_largest+1))

    for i in range(im_class_num):
        new_chosen = idx_train[(labels==(c_largest-i))[idx_train]]
        if portion == 0:#refers to even distribution
            c_portion = int(avg_number/new_chosen.shape[0])

            for j in range(c_portion):
                if chosen is None:
                    chosen = new_chosen
                else:
                    chosen = torch.cat((chosen, new_chosen), 0)

        else:
            c_portion = int(portion)
            portion_rest = portion-c_portion
            for j in range(c_portion):
                num = int(new_chosen.shape[0])
                new_chosen = new_chosen[:num]

                if chosen is None:
                    chosen = new_chosen
                else:
                    chosen = torch.cat((chosen, new_chosen), 0)
            
            num = int(new_chosen.shape[0]*portion_rest)
            new_chosen = new_chosen[:num]

            if chosen is None:
                chosen = new_chosen
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
            

    add_num = chosen.shape[0]
    new_adj = adj_back.new(torch.Size((adj_back.shape[0]+add_num, adj_back.shape[0]+add_num)))
    new_adj[:adj_back.shape[0], :adj_back.shape[0]] = adj_back[:,:]
    new_adj[adj_back.shape[0]:, :adj_back.shape[0]] = adj_back[chosen,:]
    new_adj[:adj_back.shape[0], adj_back.shape[0]:] = adj_back[:,chosen]
    new_adj[adj_back.shape[0]:, adj_back.shape[0]:] = adj_back[chosen,:][:,chosen]

    #ipdb.set_trace()
    features_append = deepcopy(features[chosen,:])
    labels_append = deepcopy(labels[chosen])
    idx_new = np.arange(adj_back.shape[0], adj_back.shape[0]+add_num)
    idx_train_append = idx_train.new(idx_new)

    features = torch.cat((features,features_append), 0)
    labels = torch.cat((labels,labels_append), 0)
    idx_train = torch.cat((idx_train,idx_train_append), 0)
    adj = new_adj.to_sparse()

    return adj, features, labels, idx_train



def src_smote(adj,features,labels,idx_train, portion=1.0, im_class_num=3):
    c_largest = labels.max().item()
    adj_back = adj.to_dense()
    chosen = None
    new_features = None

    #ipdb.set_trace()
    avg_number = int(idx_train.shape[0]/(c_largest+1))

    for i in range(im_class_num):
        new_chosen = idx_train[(labels==(c_largest-i))[idx_train]]
        if portion == 0:#refers to even distribution
            c_portion = int(avg_number/new_chosen.shape[0])

            portion_rest = (avg_number/new_chosen.shape[0]) - c_portion

        else:
            c_portion = int(portion)
            portion_rest = portion-c_portion
            
        for j in range(c_portion):
            num = int(new_chosen.shape[0])
            new_chosen = new_chosen[:num]

            chosen_embed = features[new_chosen,:]
            distance = squareform(pdist(chosen_embed.cpu().detach()))
            np.fill_diagonal(distance,distance.max()+100)

            idx_neighbor = distance.argmin(axis=-1)
            
            interp_place = random.random()
            embed = chosen_embed + (chosen_embed[idx_neighbor,:]-chosen_embed)*interp_place

            if chosen is None:
                chosen = new_chosen
                new_features = embed
            else:
                chosen = torch.cat((chosen, new_chosen), 0)
                new_features = torch.cat((new_features, embed),0)
            
        num = int(new_chosen.shape[0]*portion_rest)
        new_chosen = new_chosen[:num]

        chosen_embed = features[new_chosen,:]
        distance = squareform(pdist(chosen_embed.cpu().detach()))
        np.fill_diagonal(distance,distance.max()+100)

        idx_neighbor = distance.argmin(axis=-1)
            
        interp_place = random.random()
        embed = chosen_embed + (chosen_embed[idx_neighbor,:]-chosen_embed)*interp_place

        if chosen is None:
            chosen = new_chosen
            new_features = embed
        else:
            chosen = torch.cat((chosen, new_chosen), 0)
            new_features = torch.cat((new_features, embed),0)
            

    add_num = chosen.shape[0]
    new_adj = adj_back.new(torch.Size((adj_back.shape[0]+add_num, adj_back.shape[0]+add_num)))
    new_adj[:adj_back.shape[0], :adj_back.shape[0]] = adj_back[:,:]
    new_adj[adj_back.shape[0]:, :adj_back.shape[0]] = adj_back[chosen,:]
    new_adj[:adj_back.shape[0], adj_back.shape[0]:] = adj_back[:,chosen]
    new_adj[adj_back.shape[0]:, adj_back.shape[0]:] = adj_back[chosen,:][:,chosen]

    #ipdb.set_trace()
    features_append = deepcopy(new_features)
    labels_append = deepcopy(labels[chosen])
    idx_new = np.arange(adj_back.shape[0], adj_back.shape[0]+add_num)
    idx_train_append = idx_train.new(idx_new)

    features = torch.cat((features,features_append), 0)
    labels = torch.cat((labels,labels_append), 0)
    idx_train = torch.cat((idx_train,idx_train_append), 0)
    adj = new_adj.to_sparse()

    return adj, features, labels, idx_train

def recon_upsample(embed, labels, idx_train, adj=None, portion=1.0, im_class_num=3):
    c_largest = labels.max().item()
    avg_number = int(idx_train.shape[0]/(c_largest+1))
    #ipdb.set_trace()
    adj_new = None

    for i in range(im_class_num):
        chosen = idx_train[(labels==(c_largest-i))[idx_train]]
        num = int(chosen.shape[0]*portion)
        if portion == 0:
            c_portion = int(avg_number/chosen.shape[0])
            num = chosen.shape[0]
        else:
            c_portion = 2

        for j in range(c_portion):
            chosen = chosen[:num]

            chosen_embed = embed[chosen,:]
            distance = squareform(pdist(chosen_embed.cpu().detach()))
            np.fill_diagonal(distance,distance.max()+100)

            idx_neighbor = distance.argmin(axis=-1)
            
            interp_place = random.random()
            new_embed = embed[chosen,:] + (chosen_embed[idx_neighbor,:]-embed[chosen,:])*interp_place


            new_labels = labels.new(torch.Size((chosen.shape[0],1))).reshape(-1).fill_(c_largest-i)
            idx_new = np.arange(embed.shape[0], embed.shape[0]+chosen.shape[0])
            idx_train_append = idx_train.new(idx_new)

            embed = torch.cat((embed,new_embed), 0)
            labels = torch.cat((labels,new_labels), 0)
            idx_train = torch.cat((idx_train,idx_train_append), 0)

            if adj is not None:
                if adj_new is None:
                    adj_new = adj.new(torch.clamp_(adj[chosen,:] + adj[idx_neighbor,:], min=0.0, max = 1.0))
                else:
                    temp = adj.new(torch.clamp_(adj[chosen,:] + adj[idx_neighbor,:], min=0.0, max = 1.0))
                    adj_new = torch.cat((adj_new, temp), 0)

    if adj is not None:
        add_num = adj_new.shape[0]
        new_adj = adj.new(torch.Size((adj.shape[0]+add_num, adj.shape[0]+add_num))).fill_(0.0)
        new_adj[:adj.shape[0], :adj.shape[0]] = adj[:,:]
        new_adj[adj.shape[0]:, :adj.shape[0]] = adj_new[:,:]
        new_adj[:adj.shape[0], adj.shape[0]:] = torch.transpose(adj_new, 0, 1)[:,:]

        return embed, labels, idx_train, new_adj.detach()

    else:
        return embed, labels, idx_train

def adj_mse_loss(adj_rec, adj_tgt, adj_mask = None):
    edge_num = adj_tgt.nonzero().shape[0]
    total_num = adj_tgt.shape[0]**2

    neg_weight = edge_num / (total_num-edge_num)

    weight_matrix = adj_rec.new(adj_tgt.shape).fill_(1.0)
    weight_matrix[adj_tgt==0] = neg_weight

    loss = torch.sum(weight_matrix * (adj_rec - adj_tgt) ** 2)

    return loss



def print_edges_num(dense_adj, labels):
    c_num = labels.max().item()+1
    dense_adj = np.array(dense_adj)
    labels = np.array(labels)

    for i in range(c_num):
        for j in range(c_num):
            #ipdb.set_trace()
            row_ind = labels == i
            col_ind = labels == j

            edge_num = dense_adj[row_ind].transpose()[col_ind].sum()
            # print("edges between class {:d} and class {:d}: {:f}".format(i,j,edge_num))
