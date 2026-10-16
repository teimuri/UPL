import torch
import numpy as np
import torch_geometric as tg

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import copy
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid,Amazon,WebKB,WikipediaNetwork,AttributedGraphDataset, Flickr
from ogb.nodeproppred import PygNodePropPredDataset
import torch
from torch_geometric.data import Data
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

# Officially allow PyTorch to load PyG's graph structures
torch.serialization.add_safe_globals([
    Data,
    GlobalStorage, 
    DataEdgeAttr, 
    DataTensorAttr
])

num_threads=10
class Graph_Handler():
    def __init__(self,**settings):
        # end_time = time.perf_counter()*100000000%(2**20)
    
        
        self.settings=settings

        


        self.verbose=settings.get('verbose',0)

        self.num_threads=num_threads
    def generate_graphs(self):
        graphs=[]
        
        

            
        mint_dataset = get_dataset(self.settings['data_set'], self.settings['data_path'],**self.settings)
        if self.settings['data_set'] in ['Computers','Photo']:
        

            mint_dataset.train_mask = torch.zeros(len(mint_dataset.y),dtype=torch.bool)
            mint_dataset.val_mask = torch.zeros(len(mint_dataset.y),dtype=torch.bool)
            mint_dataset.test_mask = torch.zeros(len(mint_dataset.y),dtype=torch.bool)
            gg = []
            for class_id in mint_dataset.y.unique():

                idx = (torch.where(mint_dataset.y == class_id)[0])
                random_permutations = torch.randperm(len(idx))

                mint_dataset.train_mask[idx[random_permutations[:20]]] = 1
                mint_dataset.val_mask[idx[random_permutations[20:65]]] = 1
                mint_dataset.test_mask[idx[random_permutations[65:120]]] = 1
        elif self.settings['data_set'] in ['arxiv']:
            # When loading the dataset:
            split_idx = mint_dataset.split_idx

            # Extract the specific indices
            train_idx = split_idx["train"]
            val_idx = split_idx["valid"]
            test_idx = split_idx["test"]

            # Convert them to boolean masks for your pipeline
            num_nodes = mint_dataset.y.shape[0]

            mint_dataset.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            mint_dataset.train_mask[train_idx] = True

            mint_dataset.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            mint_dataset.val_mask[val_idx] = True

            mint_dataset.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            mint_dataset.test_mask[test_idx] = True


        for i in range(self.settings['num_runs']):
            data = copy.deepcopy(mint_dataset)
            

            
            data.train_mask,data.val_mask,data.test_mask = self.add_masks(data.clone(),index = i,**self.settings)
            total_mask = data.train_mask+data.val_mask+data.test_mask
            # if self.settings['data_set']!='arxiv':
            #     data.degree_dist = [data.adj_matrix[0,(data.y==i)*total_mask].sum(dim=0).max().item() for i in range(data.y.max().item()+1)]
            # else:
            #     data.degree_dist = [0 for i in range(data.y.max().item()+1)]
            # raise ValueError([data.adj_matrix[0,(data.y==i)*total_mask].sum(dim=0).max().item() for i in range(data.y.max().item()+1)])
            print([(data.y[data.train_mask]==i).sum() for i in range(data.y.max().item()+1)])
            graphs.append(data.clone())



        return graphs
        

        
    def pca(self,data):
        
                
        data_numpy = data.x.numpy()

        
        pca = PCA(n_components=self.settings['pca_dim'])

        # Fit the PCA model to your data
        pca.fit(data_numpy)

        # Transform your data into the lower-dimensional space
        projected_data = pca.transform(data_numpy)
        data.x = torch.tensor(projected_data,dtype=torch.float32)
        return data
    
    def lda(self,data):
        lda = LinearDiscriminantAnalysis(n_components=self.settings['lda_dim'])  # You can choose the number of components
        _ = lda.fit_transform(data.x[data.train_mask], data.y[data.train_mask])
        projected_data = lda.transform(data.x)
        data.x = torch.tensor(projected_data,dtype=torch.float32)

        return data 
    
    def add_masks(self,data,**settings):

        

        # Clip the values between the specified range
        if settings['data_set'] in ['Computers','Photo','Wiki']:
            

            train_mask = torch.zeros(len(data.y),dtype=torch.bool)
            val_mask = torch.zeros(len(data.y),dtype=torch.bool)
            test_mask = torch.zeros(len(data.y),dtype=torch.bool)
            gg = []
            for class_id in data.y.unique():

                idx = (torch.where(data.y == class_id)[0])
                random_permutations = torch.randperm(len(idx))

                train_mask[idx[random_permutations[:20]]] = 1
                val_mask[idx[random_permutations[20:65]]] = 1
                test_mask[idx[random_permutations[65:120]]] = 1

        else:
            masks = [data.train_mask.clone(),data.val_mask.clone(),data.test_mask.clone()]
            
            train_mask = masks[0]
            val_mask = masks[1]
            test_mask = masks[2]
            # Create a train mask where the indices of the training nodes are set to 1
            if settings['data_set'] in ['chameleon','squirrel','Wisconsin']:
                    
                    mask_idx = settings['index']
                    train_mask = train_mask[:,mask_idx%10]
                    val_mask = val_mask[:,mask_idx%10]
                    test_mask = test_mask[:,mask_idx%10]
        
        ########################################################
        ########There might be something wrong with this########
        # raise ValueError(settings['number_of_classes']//2)
        max_class = max([((data.y==i)*train_mask).sum() for i in range(settings['number_of_classes'] - settings['number_of_classes']//2)])
        
        for i in range(settings['number_of_classes']):
            num_train = int(max_class*settings['imbalance_map'][i])
            class_train_idx = (torch.where((data.y == i) * train_mask))[0]
            index2_remove=find_index2_remove(class_train_idx,num_train)
            train_mask[index2_remove] = False
        return [train_mask,val_mask,test_mask]

    
def trim(data,settings):

    used_nodes_index = []
    for i in range(settings['number_of_classes']):
        idx = (np.where(data.y.numpy() == i)[0])
        
        np.random.shuffle(idx)
        used_nodes_index.extend(idx[:settings['n_c'][i]])

    used_nodes_index = torch.tensor(used_nodes_index)
    used_nodes=torch.zeros_like(data.y,dtype=torch.bool)
    used_nodes[used_nodes_index] = 1
    
    node_mapping = torch.zeros_like(used_nodes,dtype=torch.long)-10

    node_mapping[used_nodes] = torch.arange(0, used_nodes.sum())
    # raise ValueError(list(node_mapping))
    node_mapping = node_mapping.unsqueeze(0)
    
    labels = data.y[used_nodes]
    
    valid_edges = used_nodes[data.edge_index[0]] & used_nodes[data.edge_index[1]]
    edge_index = data.edge_index[:, valid_edges]
    
    edge_index = torch.cat((node_mapping[:,edge_index[0,:]],node_mapping[:,edge_index[1,:]]),dim=0)
    
    edge_attr = torch.zeros_like(data.edge_index[0])
    features = data.x[used_nodes]
    
    
    
    
        
    return tg.data.Data(x=features, edge_index=edge_index, edge_attr=edge_attr, y=labels)
    
    
        
        
def get_dataset(name, path, split_type='public',**kwargs):
     

    transform = T.Compose([T.NormalizeFeatures(), T.ToUndirected()])
    if name == "Cora" or name == "CiteSeer" or name == "PubMed":
        

        dataset = Planetoid(path, name, transform=transform, split=split_type)
        
    elif name =="Computers" or name == "Photo":
        dataset = Amazon(path, name, transform=transform)

    elif name =="Wiki":
        dataset = AttributedGraphDataset(path, name, transform=transform)

    elif name == "chameleon" or name == "squirrel":
        
        dataset = WikipediaNetwork(path, name, transform = transform)
    elif name == 'arxiv':
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", transform=None, root="./data")
        split_idx = dataset.get_idx_split()

    elif name == "Flickr":
        dataset = Flickr(path, transform=transform)

    elif name == "Wisconsin":
        
        
        dataset = WebKB(path, name, transform = transform)
    else:
        raise NotImplementedError("Not Implemented Dataset!")
    dataset = dataset[0]
    
    num_nodes = dataset.x.shape[0]
    # Create a sparse boolean matrix
    dataset.adj_matrix = torch.sparse_coo_tensor(
        dataset.edge_index, 
        torch.ones(dataset.edge_index.shape[1], dtype=torch.bool), 
        (num_nodes, num_nodes)
    )





    # Unpack the source (row) and destination (column) nodes
    src, dst = dataset.edge_index
    
    # Mimic torch.tril() by keeping only edges where row >= col
    lower_triangular_mask = src >= dst
    
    # Apply the mask to get the unidirectional edges
    dataset.edge_index_unidirectional = dataset.edge_index[:, lower_triangular_mask]


    
    
    return dataset

def find_index2_remove(indexes,num_train):
    
    
    shuffled_id_index = torch.randperm(len(indexes))[num_train:]
    indexes_2remove = indexes[shuffled_id_index] 
    
    
    return indexes_2remove
    
            
            
