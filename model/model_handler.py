from typing import Any
from model.models import GNN
import numpy as np
from tqdm import tqdm
from Graph.graph_handler import *
import copy
import torch.nn.functional as functional
import torch.nn as nn
import numpy as np
import time
from torchvision.ops import sigmoid_focal_loss
import torch_geometric as tg
from model.balanced_softmax import BalancedSoftmax
from model.gens import *
from model.tam import *
from sklearn.metrics import balanced_accuracy_score, f1_score
from utils import set_seed


start_time = time.time()
pseudo_label_time = []
uncertainty_time = []
Gnn_train_time = []
class model_handler():
    def __init__(self,in_channels,out_channels,**settings):
        
        # self.model_type=globals()[model_type]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model = GNN(model=settings['model_type'],in_channels=self.in_channels,out_channels=self.out_channels,**settings).to(device)
        self.loss_type = settings['loss_type']

        self.settings = settings
        self.model.to(device)
        
        
        pass
    
    
    def __call__(self, data,pseudo_run=False,run_index=0,test_model = None) -> Any:
        global flex_match,start_time,pseudo_label_time,uncertainty_time,Gnn_train_time
        
        data = data.clone().to(device)
        
        
        
        
        
        imbalance_map=torch.tensor(self.settings['imbalance_map'])
        imbalance_map=imbalance_map<=imbalance_map.mean()
        
        if pseudo_run * (self.settings['pseudo_labeler']=='fixed'):
            out_probs = []
            if self.settings['uncertainty'] or self.settings['entropy_uncertainty']:
                start_time = time.time()    
                for _ in range(self.settings['num_augmentations']):
                    # set_seed(id)
                    num_to_remove=self.settings['nums_to_remove']
                    indices_to_keep = torch.randperm(len(data.edge_index_unidirectional[0]))[num_to_remove:]
                    my_uni_edge_index = data.edge_index_unidirectional[:,indices_to_keep].clone()
                    my_edge_index = torch.cat((my_uni_edge_index,my_uni_edge_index[[1,0]]),dim=1)
                    # raise ValueError(my_edge_index.shape)
                    out=self.model(data.x, my_edge_index,data.adj_matrix,augment = pseudo_run)
                    # raise ValueError(functional.softmax(out,dim=1)[0],torch.exp(functional.log_softmax(out,dim=1)[0]))
                    out = functional.log_softmax(out,dim=1)
                    out_probs.append(out)
                
                out_probs_stack = torch.stack(out_probs,dim=0)
                
                out_mean = torch.mean(out_probs_stack,dim=0)
                uncertainty_time.append(time.time()-start_time)
            # best_out_probs = test_model.model(data.x, data.edge_index,data.adj_matrix,augment = pseudo_run)
            
            start_time = time.time()
            out_probs = self.model(data.x, data.edge_index,data.adj_matrix,augment = pseudo_run)
            out_probs = (out_probs)
            out_probs = functional.log_softmax(out_probs,dim=1)
            if pseudo_run:
                pseudo_label_time.append(time.time()-start_time)
            else:
                Gnn_train_time.append(time.time()-start_time)
            max_value, max_idx = torch.max(out_probs, dim=1)
            if self.settings['uncertainty']:
                out_std = torch.std(out_probs_stack,dim=0)
                max_std = out_std.gather(1, max_idx.view(-1,1)).squeeze(1)
            if self.settings['entropy_uncertainty']:
                out_probs_entropy = functional.softmax(out_probs_stack,dim=0)
                out_entropy = torch.sum(-out_probs_entropy*torch.log(out_probs_entropy),dim=0)
                min_entropy,_ = torch.min(out_entropy,dim=1)

                # raise ValueError(max_entropy.unique().cpu().tolist())
            max_probs = out_probs.gather(1,max_idx.view(-1,1)).squeeze(1)
            
            
            
            # plt.hist((max_std.cpu().flatten()), bins=100,range=(0,0.1), color='skyblue', edgecolor='skyblue')
            # plt.yscale('log')
            # plt.show()
            # raise ValueError(max_std)
            # plt.clf()


            #Take the log_softmax of the output since it has not been taken yet
            pseudo_labels_mask = data.train_mask.clone()
            pseudo_labels = data.y.clone() * 0
            pseudo_labels[pseudo_labels_mask] = data.y[pseudo_labels_mask].clone()
            
            # uncertain_thr_list = self.settings['uncertain_thr']*imbalance_map
            # raise ValueError(self.settings['pseudo_labels_thr'])
            
            pseudo_labels_thr_list = self.settings['pseudo_labels_thr'][0]*imbalance_map
            
            for class_id,(imb_flag,pseudo_thr) in enumerate(zip(imbalance_map,pseudo_labels_thr_list)):
                if imb_flag:
                    pseudo_class_nodes = max_probs[max_idx==class_id]
                    
                    run_index = run_index -1
                    
                    temp_pseudo_mask = (pseudo_class_nodes>np.log(pseudo_thr))
                    if self.settings['pseudo_thr_band']!=None:
                        # raise ValueError(np.log(np.e**(pseudo_thr)+self.settings['pseudo_thr_band']))
                        temp_pseudo_mask = temp_pseudo_mask* (pseudo_class_nodes<(np.log(pseudo_thr+self.settings['pseudo_thr_band'])))
                    new_pseudos_mask = (temp_pseudo_mask*(~data.train_mask[max_idx==class_id]))
                    if self.settings['uncertainty']:
                        pseudo_class_std = max_std[max_idx==class_id]
                    
                        uncertain_thr = out_std.mean() - pseudo_class_std.mean()/5
                        uncertain_mask = (pseudo_class_std<(uncertain_thr))
                        new_pseudos_mask = new_pseudos_mask * uncertain_mask
                    if self.settings['entropy_uncertainty']:
                        pseudo_class_entropy = min_entropy[max_idx==class_id]
                        if len(pseudo_class_entropy)>0:
                            # theoritical_probs = functional.log_softmax(torch.ones([self.settings['num_augmentations']]),dim=0)
                            # theoritical_entropy = torch.sum(-torch.exp(theoritical_probs)*theoritical_probs)
                            sorted_entropy,_ = torch.sort(pseudo_class_entropy)
                            uncertain_thr = sorted_entropy[int(len(sorted_entropy)*self.settings['entropy_uncertainty_percent'])]
                            uncertain_mask = (pseudo_class_entropy>(uncertain_thr))
                            # raise ValueError(uncertain_thr,uncertain_mask.tolist())
                            new_pseudos_mask = new_pseudos_mask * uncertain_mask
                    pseudo_labels_mask[max_idx==class_id] += new_pseudos_mask

                    pseudo_labels[(max_idx==class_id)] += new_pseudos_mask*class_id
                    
            data.pseudo_labels = pseudo_labels
            data.pseudo_labels_mask = pseudo_labels_mask
            
        else:
            
            out=self.model(data.x,data.edge_index,data.adj_matrix,augment = pseudo_run)
            if self.settings['pseudo_labeler']=='flex_match':
                data.pseudo_labels,data.pseudo_labels_mask=flex_match.flexmatch_labeler(data.pseudo_labels_mask,out.clone(),mapping_func,**self.settings)
            if self.loss_type not in ['bs']:
                out = functional.log_softmax(out, dim=1)
            
            data.x = out
        
        return data

    def parameters(self):
        return self.model.reg_params,self.model.non_reg_params


def train(data_org,model_handler,**settings):
    
    global saliency, aggregator, neighbor_dist_list,flex_match,best_val_overall,best_model
    saliency, prev_out = None, None
    
    
    class_num_list=[(data_org.pseudo_labels[data_org.pseudo_labels_mask]==i).sum().item() for i\
                in range(settings['number_of_classes'])]
    # print(class_num_list)
    def val_test(data,mask):
        
        model_handler.model.eval()
        logits = model_handler.model(data.x, data.edge_index,data.adj_matrix)
        accs, baccs, f1s = [], [], []
        
        for i, mask in enumerate([mask]):
            pred = logits[mask].max(1)[1]
            y_pred = pred.cpu().numpy()
            y_true = data.y[mask].cpu().numpy()
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            bacc = balanced_accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')

            accs.append(acc)
            baccs.append(bacc)
            f1s.append(f1)

        return acc, bacc, f1
    
    data_org = data_org.to(device)
    best_val_loss = float('inf')
    # best_epoch = 0
    patience = 5  # Number of epochs to wait for improvement
    no_improvement = 0
    best_val_acc_f1 = 0
    loss_train=[]
    loss_val=[]
    reg_params,non_reg_params_list=model_handler.parameters()
    # optimizer = torch.optim.Adam(list(models_list[0].parameters())+list(models_list[1].parameters()), lr=0.01, weight_decay=5e-4)
    # optimizer = torch.optim.Adam([reg_params],settings['lr'])
   
    # optimizer = torch.optim.Adam(parameters_list, lr = settings['lr'],  weight_decay = settings['weight_decay'])
    optim_list = [dict(params=reg_params(), weight_decay=settings['wd'])]
    optim_list.extend([dict(params=non_reg_params(), weight_decay=0) for non_reg_params in non_reg_params_list])
    
    optimizer = torch.optim.Adam(optim_list,settings['lr'])
    # raise ValueError(reg_params)
    
    if settings['use_scheduler']:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                            factor = settings['schedule_factor'],
                                                            patience = settings['schedule_patience'])
    if settings['ens']:
        neighbor_dist_list = get_ins_neighbor_dist(data_org.y.size(0), data_org.edge_index, data_org.train_mask, device)
    
    aggregator = MeanAggregation()
    
    # print('first started')
    for epoch in tqdm(range(2000),leave=False):

        loss=0

        
        
        model_handler.model.train()
        optimizer.zero_grad()
        if settings['ens']:
            out = ens(model_handler.model,data_org,epoch,**settings)
        else:
            out = model_handler(data_org)

        if settings.get('pseudo_run',False):
            
            
            class_num_list=[(out.pseudo_labels[out.pseudo_labels_mask]==i).sum() for i\
                in range(settings['number_of_classes'])]
            output = adjust_output(out.x, out.edge_index, out.pseudo_labels, \
                out.pseudo_labels_mask, aggregator, class_num_list, epoch,**settings)
            loss += criterion(output, out.pseudo_labels[out.pseudo_labels_mask],**settings)
            loss += regularizer(output,out.pseudo_labels[out.pseudo_labels_mask],**settings)    
            
        else:
            class_num_list=[(out.y[out.train_mask]==i).sum() for i in range(settings['number_of_classes'])]
            output = adjust_output(out.x, out.edge_index, out.y, \
                out.train_mask, aggregator, class_num_list, epoch,**settings)
            
            loss += criterion(output, out.y[out.train_mask],**settings)
            loss += regularizer(output,out.y[out.train_mask],**settings)
        

        # print(loss)
        if loss!=loss:
            
            raise ValueError('found')
        loss_train.append((loss.item()))
        loss.backward()
        optimizer.step()
        
        
        
        # Validation
        
        val_loss=0
        data=data_org.clone()
        with torch.no_grad():
            # for model_handle in models_list:
            #     if isinstance(model_handle,model_handler):
            #         model_handle.
            model_handler.model.eval()
            
            out = model_handler(data)
                
        # out.x = functional.log_softmax(torch.log(out.x), dim=1)
        val_loss += criterion(out.x[out.val_mask], out.y[out.val_mask],**settings)

        val_loss += regularizer(out.x[out.val_mask],out.y[out.val_mask],**settings)
        
        if settings['use_scheduler']:
            scheduler.step(val_loss)


                

        loss_val.append((val_loss.item()))
        val_acc, val_bacc, val_f1 = val_test(data_org,data_org.val_mask)
        # raise ValueError(val_acc)
        
        val_acc_f1 = (val_f1)
        if val_acc_f1 > best_val_acc_f1:
            # if epoch%10==0:
            if val_acc_f1> best_val_overall:
                # print(val_acc_f1)
                best_model = copy.deepcopy(model_handler)
                best_val_overall = val_acc_f1
            best_model_local = copy.deepcopy(model_handler)
            best_val_acc_f1 = val_acc_f1

            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= settings['schedule_patience']:
                break
    
    # print(best_val_acc_f1)
    return loss_train,loss_val,best_model,best_model_local

def test(data,best_model_handler):
    data.to(device)
    
    best_model_handler.model.eval()
    out = best_model_handler(data)
    
    pred = out.x.argmax(dim=1).cpu()
    gt=data.y.clone().cpu()
    val_results = []
    test_results = []
    for resutls_list,mask in zip([val_results,test_results],[data.val_mask,data.test_mask]):
        mask = mask.cpu()
        resutls_list.extend([pred[mask],gt[mask]])
    
    return (val_results,test_results)

def regularizer(pred,gt,regularizer,regularizer_coef,**setting):
    
    unique_values = torch.unique(gt)
    class_ratio = []
    
    for val in (unique_values):
        class_ratio.append((gt == val).sum()/len(gt))
    class_ratio=torch.tensor(class_ratio).to(device)
    class_ratio = (1/class_ratio)
    
    class_ratio = class_ratio/class_ratio.sum()
    # raise ValueError(class_ratio)
    if regularizer=='kl':
        pred_mean=torch.exp(pred).mean(dim=0)
        
        temp=regularizer_coef*(pred_mean*torch.log((pred_mean+1e-7)*len(pred_mean)))
        return temp.sum()
    elif regularizer=='kl_reverse':
        pred_mean=torch.exp(pred).mean(dim=0)
        
        temp=regularizer_coef*(pred_mean*torch.log((pred_mean+1e-7)/(class_ratio**setting["kl_power"])))
        return temp.sum()
    elif regularizer==None:
        return 0
    else:
        raise ValueError('Regularizer not found')

def criterion(pred,label,loss_type='normal',**settings):
    
    class_num_list=[(label==i).sum() for i in range(settings['number_of_classes'])]

    if loss_type =='normal':

        # loss=nn.functional.nll_loss(pred, label)
        loss = functional.cross_entropy(pred, label, weight=settings.get('weight',None), reduction=settings.get('reduction','mean'))
        
        return loss
    elif loss_type=='bs':
        loss = BalancedSoftmax(pred, label,class_num_list)
        return loss
    else:
        if loss_type =='focal':
            gamma=settings['focal_gamma']

            # Convert labels to one-hot encoding
            one_hot_labels = torch.nn.functional.one_hot(label, settings['number_of_classes']).to(torch.float)

            # print(focal_loss)
            focal_loss = sigmoid_focal_loss(pred, one_hot_labels, gamma=gamma, reduction='mean')
            return focal_loss
        ######################
        # Add terverosky loss#
        ######################
        elif loss_type=="tilted":
            t=settings['tea']
            tilted_loss=torch.log(torch.mean(torch.exp(nn.functional.nll_loss(pred, label)*t)))/t
            return tilted_loss
#Made for Graphens
def backward_hook(module, grad_input, grad_output):
    global saliency
    saliency = grad_input[0].data

def tensor_hook(grad):
    global saliency
    saliency = grad.data

def ens(model,data,epoch,**kwargs):
    global saliency,aggregator,neighbor_dist_list,prev_out
    data = data.clone()
    model.conv1.temp_weight.register_full_backward_hook(backward_hook)
    
    class_num_list=[(data.y[data.train_mask]==i).sum().item() for i in range(kwargs['number_of_classes'])]
    idx_info = [torch.where((data.y==i)*(data.train_mask))[0] for i in range(kwargs['number_of_classes'])]
    # Sampling source and destination nodes
    
    
    sampling_src_idx, sampling_dst_idx = sampling_idx_individual_dst(class_num_list, idx_info, "cpu")
    beta = torch.distributions.beta.Beta(2, 2)
    lam = beta.sample((len(sampling_src_idx),) ).unsqueeze(1)
    ori_saliency = saliency[:data.x.shape[0]] if (saliency != None) else None
    # Augment nodes
    if epoch > kwargs['warm_up']:
        with torch.no_grad():
            
            
            prev_out = aggregator(prev_out, data.edge_index)
            
            prev_out = functional.softmax(prev_out / kwargs['pred_temp'], dim=1).detach().clone()
        new_edge_index, dist_kl = neighbor_sampling(data.x.size(0), data.edge_index, sampling_src_idx, sampling_dst_idx,
                                    neighbor_dist_list, prev_out)
        new_x = saliency_mixup(data.x, sampling_src_idx, sampling_dst_idx, lam, ori_saliency, dist_kl = dist_kl, keep_prob=kwargs["keep_prob"])
    else:
        new_edge_index = duplicate_neighbor(data.x.size(0), data.edge_index, sampling_src_idx)
        
        dist_kl, ori_saliency = None, None
        new_x = saliency_mixup(data.x, sampling_src_idx, sampling_dst_idx, lam, ori_saliency, dist_kl = dist_kl)
    new_x.requires_grad = True           

    output = model(new_x, new_edge_index)
    
    prev_out = (output[:data.x.size(0)]).detach().clone() # logit propagation
    ## Train_mask modification ##
    add_num = output.shape[0] - data.train_mask.shape[0]
    new_train_mask = torch.ones(add_num, dtype=torch.bool, device= data.x.device)
    new_train_mask = torch.cat((data.train_mask, new_train_mask), dim =0)

    ## Label modification ##
    _new_y = data.y[sampling_src_idx].clone()
    new_y = torch.cat((data.y[data.train_mask], _new_y),dim =0)

    data.y = torch.cat((data.y,_new_y),dim =0)
    
    # output = adjust_output(args, output, new_edge_index, torch.cat((data.y,_new_y),dim =0), \
    #     new_train_mask, aggregator, class_num_list, epoch)
    data.x = output
    data.edge_index = new_edge_index
    data.train_mask = new_train_mask
    return data

def mk_models_table(graphs,**settings):
    global device, Bsoftmax,flex_match,best_val_overall
    
    device = settings['device']
    num_run=settings['num_runs']
    label = graphs[0].y[graphs[0].train_mask]
    # raise ValueError(label)
    # class_num_list=[(label==i).sum() for i in range(settings['number_of_classes'])]
    
    
    
    
    
    
    #dangerus
    # for data_org in (graphs):
    #     dataset=[]
    #     dataset.append(copy.deepcopy(data_org.to(device)))

    all_results = []

    
    seed =100
    ratio_list=torch.zeros((0,1))

    for data_org in tqdm(graphs):
        # print('##################NEW#####################\n#########################################')
        best_val_overall = 0
        # print(seed)
        set_seed(seed)
        seed+=1
        
        # flex_match = pseudo_labeler(data_org,**settings)
        # temp_confusions=torch.zeros((1,4*graph.y.unique().shape[0])) 
        # del dataset
        # data_edges_list=[]
        # data_features_list=[]
        # data_edges_list.append(copy.deepcopy(data_org.edge_index))
        # data_features_list.append(copy.deepcopy(data_org.x))

        
        # data_org.x = get_embedding(data_org.clone(),**settings)
        # raise ValueError(data_org.x.shape)
        data_org.pseudo_labels = torch.zeros_like(data_org.y)
        data_org.pseudo_labels_mask = torch.zeros_like(data_org.train_mask)
        gnn=model_handler(in_channels=data_org.x.shape[1],out_channels=len(data_org.y.unique()),**settings)
        loss_train,loss_val,best_model_handler,best_model_local = train(data_org.clone(),**settings,model_handler=gnn)
        
        data_pseudo = data_org.clone()
        #This to prevent any types of leaks
        data_pseudo.pseudo_labels = torch.zeros_like(data_pseudo.y)
        data_pseudo.pseudo_labels[data_pseudo.train_mask] = data_org.y[data_pseudo.train_mask].clone()
        data_pseudo.pseudo_labels_mask = data_pseudo.train_mask.clone()
        
        for run_index in tqdm(range(settings['num_extra_runs']),leave=False):
            with torch.no_grad():
                
                    
                data_pseudo = best_model_local(data_pseudo,pseudo_run = True,run_index=run_index,test_model = best_model_handler)
                    
                # raise ValueError(data_pseudo.x.shape)
                
                # link_generate_data = best_model_local(data_pseudo.clone())
                # data_pseudo.edge_index = linker(link_generate_data)
            
            # raise ValueError(linker)
            # print(data_pseudo.pseudo_labels_mask.sum(),data_pseudo.train_mask.sum())
            del gnn
            
            # set_seed(seed)
            # seed+=1
            
            gnn=model_handler(in_channels=data_pseudo.x.shape[1],out_channels=len(data_pseudo.y.unique()),**settings)
            loss_train,loss_val,best_model_handler,best_model_local = train(data_pseudo,**settings,model_handler=gnn,pseudo_run=True)
            
            
        losses_list = [loss_train,loss_val]
        results = test(data_org,best_model_handler)
        
        all_results.append([results,data_org.cpu().clone()])
                
                    

    # print(f"Pseudo: The mean is {np.mean(pseudo_label_time)} for {pseudo_label_time}")
    # print(f"Uncertainty: The mean is {np.mean(uncertainty_time)} for {uncertainty_time}")
    # print(f"Gnn_train: The mean is {np.mean(Gnn_train_time)} for {Gnn_train_time}")
    return all_results,ratio_list,losses_list
from collections import defaultdict, Counter
def mapping_func(mode='convex'):
    """Returnt he Mapping func."""
    
    if mode == 'convex':
        return lambda x:x / (2 - x)
class pseudo_labeler():
    def __init__(self,data,**settings):
        self.cls_thresholds=settings['pseudo_labels_thr']
        self.learning_status = [-1] * data.train_mask.sum()
        self.num_classes = settings['number_of_classes']
        self.mapping = mapping_func(settings['mapping'])
        
    def flexmatch_labeler(self,labled_mask,probs,mapping,**settings):
        # supervised loss
        probs = functional.softmax(probs, dim=1)
        
        counter = Counter(self.learning_status)
        # normalize the status
        num_unused = counter[-1]
        num_unlabeled = (~labled_mask).sum()
        if num_unused != num_unlabeled:
            max_counter = max([counter[c] for c in range(self.num_classes)])
            if max_counter < num_unused:
                # normalize with eq.11
                sum_counter = sum([counter[c] for c in range(self.num_classes)])
                denominator = max(max_counter, num_unlabeled - sum_counter)
            else:
                denominator = max_counter
            # threshold per class
            for c in range(self.num_classes):
                beta = counter[c] / denominator
                raise ValueError(self.mapping(beta))
                self.cls_thresholds[c] = mapping(beta) * settings['pseudo_labels_thr']

        # update the pseudo label
        with torch.no_grad():
            
            max_prob, hard_label = torch.max(probs[~labled_mask], dim=1)
            
            over_threshold = max_prob > settings['pseudo_labels_thr']
            if over_threshold.any():
                u_i = u_i.to(device)
                sample_index = u_i[over_threshold].tolist()
                pseudo_label = hard_label[over_threshold].tolist()
                for i, l in zip(sample_index, pseudo_label):
                    self.learning_status[i] = l

        # unsupervised loss
        batch_threshold = torch.index_select(self.cls_thresholds, 0, hard_label)
        
        indicator = max_prob > batch_threshold 
        pseudo_labels = torch.zeros_like(labled_mask)
        pseudo_labels_mask = torch.zeros_like(labled_mask)
        pseudo_labels[~labled_mask] = hard_label
        pseudo_labels_mask[~labled_mask] =  indicator
        
        return pseudo_labels,pseudo_labels_mask
        lu = (criterion(us_pred, hard_label) * indicator).mean()
        total_loss += lu * unsupervised_weight