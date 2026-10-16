# train.py
import wandb
import torch
import numpy as np
import random # for demo script
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch_geometric.utils import degree

class log_wandb():
    def __init__(self,wandb_profile,project_name,**settings) -> None:
        

        # self.model_names=settings['model']
        
        self.project_name=project_name
        self.wandb_profile = wandb_profile
        self.wandb_path = self.wandb_profile + '/' + self.project_name
        self.settings=settings.copy()
        
        
        self.num_runs=settings['num_runs']


        
        self.ratio_lists=[]
        self.counter=0
        self.run_losses=[]
        if self.settings['debug']==0:
            wandb_settings={}
            for key in self.settings.keys():
                
                
                if key=='debug' or key=='device':
                    continue
                elif (type(self.settings[key]) is torch.Tensor):
                    try:
                        for id,value in enumerate(self.settings[key]):
                            
                            wandb_settings[f'{key}_{id}']=value
                        wandb_settings[key]=self.settings[key]
                    except:
                        wandb_settings[key]=self.settings[key]
                elif (type(self.settings[key]) is list) and (self.settings[key][0] is not str):
                    wandb_settings[key]=self.settings[key]
                    for id,value in enumerate(self.settings[key]):
                        wandb_settings[f'{key}_{id}']=value
                else:
                    wandb_settings[key]=self.settings[key]
        self.accs={'validation':[],'test':[]}
        self.baccs={'validation':[],'test':[]}
        self.f1s={'validation':[],'test':[]}
        self.raw_results = {'validation':[],'test':[]}
        self.degree_log = []
            
        
    def log(self,data,ratio_list,losses_list):
        for (val_result,test_result),data_org in data:
            total_mask = data_org.train_mask+data_org.val_mask+data_org.test_mask
            

            # 1. Calculate the degree of all nodes (assuming edge_index[0] is the source node)
            all_degrees = degree(data_org.edge_index[0], num_nodes=data_org.num_nodes)

            # 2. Iterate through classes and find the max degree for the masked nodes
            self.degree_log.append([])
            for i in range(data_org.y.max().item() + 1):
                # Create the mask for class 'i' and the total_mask
                mask = (data_org.y == i) * total_mask
                
                # Apply the mask to the degrees and find the max
                if mask.any():
                    max_deg = all_degrees[mask].max().item()
                else:
                    max_deg = 0.0 # Fallback if no nodes match the mask
                    
                self.degree_log[-1].append(max_deg)            
            for result_name,(run_pred,run_gt)  in zip(('validation','test'),(val_result,test_result)):
                
                acc = run_pred.eq(run_gt).sum().item() / len(run_gt)
                bacc = balanced_accuracy_score(run_gt, run_pred)
                f1 = f1_score(run_gt, run_pred, average='macro')
                self.accs[result_name].append(acc)
                self.baccs[result_name].append(bacc)
                self.f1s[result_name].append(f1)
                self.raw_results[result_name].append([run_pred,run_gt])
                

        # accs.append(acc)
        # baccs.append(bacc)
        # f1s.append(f1)
        self.run_losses.append(losses_list)
       
    def push(self,run):
        # raise ValueError(torch.tensor(self.f1s['validation']).mean(),torch.tensor(self.f1s['test']).mean())
        # raise ValueError(torch.tensor(self.f1s['test']).mean())
        results={}
        for score_type,score_dict in zip(('acc','bacc','f1','raw'),(self.accs,self.baccs,self.f1s,self.raw_results)):
            print(score_type)
            for key,value in score_dict.items():
                
                results[f'{score_type}_{key}']=value
                if score_type!='raw':
                    results[f'{score_type}_{key},mean']=np.mean(value)
                    results[f'{score_type}_{key},var']=np.std(value)/np.sqrt(len(value))
        
        if self.settings['debug']==0:
            run.log(results)
            
            # for run_id,run_loss in enumerate(self.run_losses):
            run_loss=self.run_losses[0]
            
            for step,value in enumerate(run_loss[0]):
                
                run.log({
                    'train_loss':value,
                    'time_step': step
                })
            for step,value in enumerate(run_loss[1]):
                
                run.log({
                    'val_loss':value,
                    'time_step': step
                })
            

            run.log_code()
            run.finish()
        else:
            self.degree_log
            with open(f"Results/{self.project_name}.md", "a") as file:
                file.write(f'{self.settings["run_name"]}:')
                if self.settings['classes_to_keep']!=None:
                    file.write('\n<!--')
                    file.write(f'\ndegree_log:')
                    file.write(f'\n{self.settings["classes_to_keep"]}')
                    for log in self.degree_log:
                        file.write(f'\n{log}')
                else:
                    file.write('\n<!--')
                file.write(f'\nf1_validation: {[f"{value*100:.2f}" for value in results["f1_validation"]]}')
                file.write(f'\nf1_validation,mean: {(results["f1_validation,mean"]*100):.2f}')
                file.write(f'\nf1_validation,var: {(results["f1_validation,var"]*100):.2f}')
                file.write(f'\nacc_test: {[f"{value*100:.2f}" for value in results["acc_test"]]}')

                file.write(f'\nbacc_test: {[f"{value*100:.2f}" for value in results["bacc_test"]]}')
                file.write(f'\nf1_test: {[f"{value*100:.2f}" for value in results["f1_test"]]}')
                file.write(f'\n& {(results["acc_test,mean"]*100):.2f} \\tiny{{$\\pm {(results["acc_test,var"]*100):.2f}$}}\n')
                file.write('-->\n')
                #output for markdown
                # file.write(f'\n{(results["bacc_test,mean"]*100):.2f}<small>±{(results["bacc_test,var"]*100):.2f}</small> ')
                # file.write(f'& {(results["f1_test,mean"]*100):.2f}<small>±{(results["f1_test,var"]*100):.2f}</small> ')
                #Output for latex
                file.write(f'\n& {(results["bacc_test,mean"]*100):.2f} \\tiny{{$\\pm {(results["bacc_test,var"]*100):.2f}$}} ')
                file.write(f'& {(results["f1_test,mean"]*100):.2f} \\tiny{{$\\pm {(results["f1_test,var"]*100):.2f}$}}\n\n\n\n')


                # file.write(f'\nf1_test,mean: ')
                # file.write(f'\nf1_test,var: {(results["f1_test,var"]*100):.2f}\n\n')
                
            # print('f1_validation: ',results['f1_validation,mean'],'\nf1_test:',results['f1_test,mean'],'\nf1_test_var:',results['f1_test,var'])
            


