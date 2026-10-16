
import torch
from model.model_handler import *
from Graph.graph_handler import *

from wandb_handler import *

import utils
from args import get_args

def graph_thread(**settings):

    Graph_instance = Graph_Handler(**settings)
    graphs=Graph_instance.generate_graphs()
    return graphs

def model_thread(graphs,settings):


    all_results,ratio_list,losses_list=mk_models_table(graphs,**settings)
    
    logger.log(all_results,ratio_list,losses_list)

torch.cuda.empty_cache()
seed =50
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(seed)
np.random.seed(seed)
args = get_args()

number_of_classes_guide={'Cora':7,'Wisconsin':5,'chameleon':5,'squirrel':5,'PubMed':3,'CiteSeer':6}


#Just to start the process in the gpu
torch.tensor(1).to("cuda:0")





    


confusion_dfs={}
# raise ValueError(settings['wandb_profile'])





def main(config=None):
    global logger,settings
    if args.debug==False:
        #This means we are not in a sweep
        run_extra_conf = {}
        if args.project_name!=None:
            run_extra_conf.update({'project':args.project_name})
        if args.run_name!=None:
            run_extra_conf.update({'name':args.run_name})
        run = wandb.init(config=config,**run_extra_conf)
            
        
        

        settings = utils.make_settings(args,run.config)
        run.config.update(settings)
        logger=log_wandb(**settings)
    else:
        settings = utils.make_settings(args,{})
        logger=log_wandb(**settings)
    utils.Error_check(settings)

        
    graphs=graph_thread(**settings)
    
    model_thread(graphs,settings)
    if settings['debug'] ==0:
        logger.push(run)
    else:
        logger.push('Wandb has not been used.')
if __name__ == '__main__':
    # args.debug=True
    main(config={})

