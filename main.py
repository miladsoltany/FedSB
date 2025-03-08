from mixstyle.mixstyle import resnet50_fc512_ms12_a0d1_domprior, resnet18_fc512_ms12
from domainbed import datasets
from domainbed import hparams_registry
import argparse 
import torch
from domainbed.lib import misc
from tqdm import tqdm
import copy
import gc
import numpy as np
from torchvision.models import resnet
from torch import nn
import wandb
from datetime import datetime
import os
import json
from time import time
from itertools import cycle
TQDM_FLAG = False


import torch
import torch.nn.functional as F

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        log_probs = F.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
    
def train_source(args, hparams, da_phase, model, criterion: torch.nn.Module, train_dl, global_prototypes = None, moon_previous_nets = None):
    global device
    model.to(device)
    running_loss = 0.0
    running_prox_loss = 0.0

    lr = hparams["lr"] if da_phase=='source' else hparams["lr"] * args.lr_ratio
    
    optimizer = torch.optim.Adam(model.parameters(), lr= lr , weight_decay=args.weight_decay) 

    model.train()
    num_epochs = args.num_source_epochs if da_phase == 'source' else args.num_target_epochs
    iter_count = 0
    
    if args.fedprox_mu :
        initial_model = copy.deepcopy(model) # the model at the begining of the communication round (for fedprox and fedmoon)
    
    for epoch in range(num_epochs):

        y_true_list = list()
        num_batches = 0
        total_images = 0
        if args.budget is None:
            budget = len(train_dl)
        else:
            budget = args.budget
            
        print('Budget: ', budget)
        with tqdm(cycle(train_dl), unit="batch", disable=not TQDM_FLAG) as tepoch:
            for (imgs, labels) in tepoch:
                if budget is not None:
                    if iter_count == int(budget):
                        # print('break')
                        break
                iter_count += 1
                
                total_images += len(imgs)
                num_batches += 1
                tepoch.set_description(f"Epoch {epoch}")
                inputs = imgs.to(device)
                labels = labels.long().to(device)
                optimizer.zero_grad()

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.detach().cpu().item()
                
                
                # added fedprox proximity term
                if args.fedprox_mu:
                    proximity_term = 0
                    for w0, w1 in zip(model.parameters(), initial_model.parameters()):
                        proximity_term += (w0 - w1).norm(2) * args.fedprox_mu 
                    loss += proximity_term
                    running_prox_loss += proximity_term.detach().cpu().item()

                    
              
                tepoch.set_postfix(loss=loss.item())
                for i in range(len(outputs)):
                    y_true_list.append(labels[i].cpu().data.tolist())

                # Backward pass
                loss.backward()
                optimizer.step()
                # break
            
            
    print(f'Client Losses || CE: {running_loss * 1000 / total_images :0.5f} || Prox:  {running_prox_loss * 1000 / total_images:0.5f} --> all / 1000')


    return model, None


def federated_averaging(args, global_model, client_models, domain_weights, client_prototypes = None, num_classes = 7):
    
    if domain_weights is None:
        domain_weights = np.ones(len(client_models)) / len(client_models)
    global_state_dict = global_model.state_dict()

    for k in global_state_dict.keys():
        weighted_sum = sum(client_models[domain].state_dict()[k].float() * domain_weights[domain] 
                           for domain in range(len(client_models)))
        global_state_dict[k] = weighted_sum
    global_model.load_state_dict(global_state_dict)

    if client_prototypes is not None:
        global_prototypes = {}
        for cls in range(num_classes):
            cls = f'cls_{cls}'
            weighted_sum = sum(client_prototypes[domain][cls].float() * domain_weights[domain] 
                                for domain in range(len(client_prototypes)))
            
            global_prototypes[cls] = weighted_sum
    else:
        global_prototypes = None
        

    return global_model, global_prototypes


def test(args: argparse.Namespace, model: torch.nn.Module,
         criterion: torch.nn.Module, test_loader: torch.utils.data.DataLoader, verbose = True):
    global device
    model.to(device)
    model.eval()
    trial_results = dict()
    running_loss = 0.0
    running_corrects = 0
    y_true_list = list()
    y_pred_list = list()

    # Iterate over dataloader
    for (imgs, labels) in test_loader:
        inputs = imgs.to(device)
        labels = labels.long().to(device)

        # Forward pass
        with torch.no_grad():
            if args.use_mixstyle:
                outputs = model.classifier(model(inputs))
            else:
                outputs = model(inputs)
            loss = criterion(outputs, labels)

            for i in range(len(outputs)):
                y_true_list.append(labels[i].cpu().data.tolist())

            # Keep track of performance metrics
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs.data, 1)
            
            running_corrects += torch.sum(preds == labels.data).item()
            # print(preds)

    test_loss = running_loss / len(y_true_list)
    test_acc = float(running_corrects) / len(y_true_list)

    if verbose:
        print('Test Loss: {:.4f} Acc: {:.4f}'.format(
            test_loss, test_acc), flush=True)

    return (test_loss, test_acc, None) 



def main(args):
    if args.wandb is not None:
        wandb.init(project='FedSmoothBoost',name=args.wandb, config=args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hparams = hparams_registry.default_hparams('fedgp', 'PACS')
    dataset = vars(datasets)[args.dataset](args.dataroot, args.test_envs ,  hparams)
    clients_dls = {'train':[], 'test':[]}
    server_dls = {'train':[], 'test':[]}
    clients = []
    server = []

    for env_i, env in enumerate(dataset):

        if env_i in args.test_envs:
            server_dls['test'].append(torch.utils.data.DataLoader(
            env,
            num_workers=args.n_workers,
            batch_size=args.target_batch_size))
        else:
            clients.append(dataset.ENVIRONMENTS[env_i])
            in_, out = misc.split_dataset(env,
                int(len(env)*args.validation_fraction),
                misc.seed_hash(args.trial_seed, env_i))
            clients_dls['train'].append(torch.utils.data.DataLoader(
            out,
            num_workers=args.n_workers,
            batch_size=args.client_batch_size))
            clients_dls['test'].append(torch.utils.data.DataLoader(
            in_,
            num_workers=args.n_workers,
            batch_size=args.client_batch_size))
            
    if args.domain_weights == 'fedavg':
        client_weights = []
        for dl in clients_dls['train']:
            client_weights.append(dl.dataset.__len__())
        client_weights = np.array(client_weights)
        client_weights = client_weights / np.sum(client_weights)
    elif args.domain_weights == 'normal':
        client_weights = None
    
         
    if args.backbone == 'resnet50':
        if args.use_mixstyle:
            global_model = resnet50_fc512_ms12_a0d1_domprior(dataset.num_classes, mix = 'random')
        else:
            global_model = resnet.resnet50(weights = resnet.ResNet50_Weights.IMAGENET1K_V2)
            global_model.fc = nn.Linear(global_model.fc.in_features, dataset.num_classes)
        
    elif args.backbone == 'resnet18':
        if args.use_mixstyle:
            global_model = resnet18_fc512_ms12(dataset.num_classes, mix = 'random')
        else:
            global_model = resnet.resnet18(weights = resnet.ResNet18_Weights.IMAGENET1K_V1)
            global_model.fc = nn.Linear(global_model.fc.in_features, dataset.num_classes)
            
    elif args.backbone == 'resnet34':
        global_model = resnet.resnet34(weights = resnet.ResNet34_Weights.IMAGENET1K_V1)
        global_model.fc = nn.Linear(global_model.fc.in_features, dataset.num_classes)
    elif args.backbone == 'resnet101':
        global_model = resnet.resnet101(weights = resnet.ResNet101_Weights.IMAGENET1K_V1)
        global_model.fc = nn.Linear(global_model.fc.in_features, dataset.num_classes)
    elif args.backbone =='vit_b_32':
        import torchvision
        global_model = torchvision.models.vit_b_32(weights = torchvision.models.ViT_B_32_Weights.DEFAULT)
        global_model.heads.head = nn.Linear(global_model.heads.head.in_features, dataset.num_classes)
    elif args.backbone =='vit_b_16':
        import torchvision
        global_model = torchvision.models.vit_b_16(weights = torchvision.models.ViT_B_16_Weights.DEFAULT)
        global_model.heads.head = nn.Linear(global_model.heads.head.in_features, dataset.num_classes)

        
        
    

    global_model.to(device)
    if args.ls_eps > 0:
        print('Using Label Smoothing')
        criterion = LabelSmoothingCrossEntropy(smoothing = args.ls_eps).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss().to(device)
    
    local_models = [copy.deepcopy(global_model) for _ in clients]
    val_losses = []
    val_accuracies = []

    target_losses = []
    target_accuracies = []
    global_prototypes = None
    for epoch in range(args.num_global_epochs):
        current_epoch_prototype = {}
        t1 = time()
        da_phase = 'source'
        source_vals_accs = []
        source_vals_losses = []
        for idx, (model, train_dl, val_dl) in enumerate(zip(local_models, clients_dls['train'], clients_dls['test'])):
            model, client_prototypes = train_source(args, hparams, da_phase, model, criterion, train_dl, global_prototypes)
            (loss, acc, auc) = test(args, model, criterion, val_dl, verbose = False)
            source_vals_accs.append(acc)
            source_vals_losses.append(loss) 
            
        global_model, global_prototypes = federated_averaging(args, global_model, local_models, None, None, dataset.num_classes)
        
        for client in local_models:
            client.load_state_dict(global_model.state_dict())
        
        (loss, acc, auc) = test(args, global_model, criterion, server_dls['test'][0], verbose=False)
        
        source_vals_accs = np.round(np.mean(source_vals_accs), 4)
        source_vals_losses = np.round(np.mean(source_vals_losses), 4)
        val_losses.append(source_vals_losses)
        val_accuracies.append(source_vals_accs)
        
        target_losses.append(loss)
        target_accuracies.append(acc)
        if args.wandb is not None:
            wandb.log({'target_test_loss': loss, 'target_test_acc': acc}, step = epoch)
            wandb.log({'source_loss': source_vals_losses, 'source_acc': source_vals_accs}, step = epoch)
        
        print(f'Epoch {epoch}/{args.num_global_epochs}')
        print(f'local acc: {source_vals_accs} || local loss: {source_vals_losses}')
        print(f'target test acc: {acc:0.3f} || target test loss: {loss:0.3f}')
        t2 = time()
        print(f'Elapsed time: {round(t2 - t1, 2) }s')
        print('####################################################################################################')

    logs = {}
    logs['target_losses'] = target_losses
    logs['target_accuracies'] = target_accuracies
    logs['val_losses'] = val_losses
    logs['val_accuracies'] = val_accuracies
    
    logs_file = open(os.path.join(args.model_save_path , 'results.json') , 'w')
    json.dump(logs, logs_file)
    logs_file.close()
    
    val_loss_best_idx = np.argmin(val_losses)
    val_acc_best_idx = np.argmax(val_accuracies)
    
    print('Best Target Acc based on val loss: ', target_accuracies[val_loss_best_idx])
    print('Best Target Acc based on val acc: ', target_accuracies[val_acc_best_idx])

    
    torch.save(global_model.state_dict(), os.path.join(args.model_save_path , 'final_global_model.pt'))

    if args.wandb is not None:
        wandb.log({'TAcc_val_loss': target_accuracies[val_loss_best_idx], 'TAcc_val_acc': target_accuracies[val_acc_best_idx]})
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0]) 
    parser.add_argument('--validation_fraction', type=float, default=0.1)
    
    parser.add_argument('--trial_seed', type=int, default=1,
            help='Trial number (used for seeding split_dataset and '
            'random_hparams).')
    parser.add_argument('--num_source_epochs', type=int, default=1)
    parser.add_argument('--num_target_epochs', type=int, default=1)
    parser.add_argument('--num_global_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--client_batch_size', type=int, default=64)
    parser.add_argument('--target_batch_size', type=int, default=64)
    parser.add_argument('--dataset', type=str, default='PACS')
    parser.add_argument('--dataroot', type=str, default='../data/')
    parser.add_argument('--model_save_path', type=str, default=None)
    parser.add_argument('--wandb', type=str, default=None)
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--task', type=str, default='FDG')
    parser.add_argument('--n_workers', type = int, default = 2)
    parser.add_argument('--proto_weight', type=float, default=0.0)
    parser.add_argument('--domain_weights', type=str, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--proto_ema_coef', type=float, default=1.0)
    parser.add_argument('--budget', type=int, default=None) 
    parser.add_argument('--ls_eps', type=float, default=0.0)
    parser.add_argument('--fedprox_mu', type=float, default=0.0)
    parser.add_argument('--use_mixstyle', action='store_true')
        

    
    args = parser.parse_args()
    
    workdir = str(datetime.now()).replace(" ", '').replace(":", '').replace("-", '').replace(".", "_")
    workdir = '_'.join((workdir , args.dataset.upper(), str(args.test_envs[0]) , args.task))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n = 0
    main_workdir = workdir + f'_{n}'
    while os.path.isdir(workdir):
        main_workdir = workdir + str(n)
        n = n + 1
    workdir = main_workdir
    
    print('workdir: ', workdir)
    
    if args.model_save_path is None:
        workdir = os.path.join('workdirs_dadg', workdir)
    else:
        workdir = os.path.join(args.model_save_path, 'workdirs', workdir)
    os.makedirs(workdir, exist_ok=True)

    cfg_file = open(os.path.join(workdir , 'cfg.json'), 'w')
    json.dump(args.__dict__, cfg_file)
    cfg_file.close()
    args.model_save_path = workdir
    
    print('Arguments =')
    for arg in vars(args):
        print('\t'+arg+':',getattr(args,arg))
    main(args)