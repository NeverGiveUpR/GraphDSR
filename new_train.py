import torch as th
import utils
import numpy as np
import time
import sympy
from Graph import GraphNet
from sklearn.metrics import r2_score
import os

def inner_training(net, X, y, max_epoch=1000, verbose=False):
    # early stop is used
    X = th.tensor(X)
    y = th.tensor(y)
    #print("X:{}, y:{}".format(X.shape, y.shape))
    patience = 20
    min_delta = 0.0001 
    epochs_no_improve = 0
    best_loss = 100000000000
    params = []
    for v in net.V:
        params += [v.weight]
    optim = th.optim.Adam(params, lr=0.1)
    loss_func = th.nn.MSELoss()

    for t in range(max_epoch):
        preds = net(X)
        loss = loss_func(y, preds.squeeze())

        if th.isnan(loss).item():
            return net, loss
        if t%10 == 0 and verbose:
            print(t, loss.item())
        if best_loss - loss > min_delta:
            best_loss = loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if loss < best_loss:
            best_loss = loss
        
        if epochs_no_improve == patience:
            if verbose:
                print(t, loss.item())
                print('Early stopping!')
            break 
        try:
            loss.backward()
            optim.step()
            optim.zero_grad()
        except:
            loss = th.tensor(100000)

    return net, loss.item()

def train(
        graphs,
        gnn,
        operators,
        args,
        X = None,
        y = None,
        percent = 0.2,
        optimizer = 'adam',
        buffer = True,
        with_const = True,
        expr = None
):
    if (optimizer == 'adam'):
        optim = th.optim.Adam(gnn.parameters(), lr=args.learning_rate)
    else:
        optim = th.optim.RMSprop(gnn.parameters(), lr=args.learning_rate)

    global_opt = {'expr':'', 'reward':-1000000, 'complex': 100000000}
    epoch_opt = {'expr':'', 'reward': -1000000, 'complex': 100000000}
    early_stop = False
    
    save_file_path = './new_results/jsonn{}'.format(str(args.node_num))
    if not os.path.exists(save_file_path):
        os.makedirs(save_file_path)
    s = time.time()
    all_time = []
    best_net = None
    for t in range(args.epoch):
        with th.autograd.set_detect_anomaly(True):
            print("----------------------------{}-{}-----------------------------".format(t, expr))
            # sample expr graphs
            #c_s = time.time()
            nodes, adjs, masks, logits_A, logits_X = graphs.sample_valid_expr_graph_(gnn)
            #c_e = time.time()
            #print("sample expression spends {}s".format(c_e-c_s))
            masks = th.cat([th.ones((args.batch_size, 1)), masks[:, :-1]],dim=1)
            logits_X = th.multiply(logits_X, masks)
            logits_A = th.multiply(logits_A, masks[:, 1:])

            # use buffer to reduce training time
            
            rewards = []
            infixes = []
            #c_s = time.time()
            inner_train_time = []
            reward_time = []
            # calculate reward through expr
            for node, adj in zip(nodes, adjs):
                net = GraphNet(node, adj, operators)
                inner_s = time.time()
                net, loss = inner_training(net, X, y)
                inner_e = time.time()
                inner_train_time.append(inner_e-inner_s)
                
                # calculate reward
                y_pred = net(th.tensor(X))
                inner_s = time.time()
                r = reward(y, y_pred.squeeze().detach().numpy())
                inner_e = time.time()
                reward_time.append(inner_e-inner_s)
                
                infix = net.to_infix()
                if r > global_opt['reward']:
                    global_opt['reward'] = r
                    global_opt['complex'] = th.sum(adj).item()
                    global_opt['expr'] = infix
                    best_net = net
                    best_net.save_as_json(expr, save_file_path)
                #print("infix:", infix)
                infixes.append(infix)
                rewards.append(r)
                # early stop
                if r>=args.stop_criteria or r>0.99:
                    print("Early stopped!")
                    global_opt['complex'] = th.sum(adj).item()
                    global_opt['reward'] = r
                    global_opt['expr'] = infix
                    e = time.time()
                    # print("time_cost:", e-s)
                    early_stop = True
                    break
            #print("inner training of one expression spends {}s".format(sum(inner_train_time)))
            #print("reward computing of one expression spends {}s".format(sum(reward_time)))
            if early_stop:
                break
            #c_e = time.time()
            #print("all constant optimizing spends {}s".format(c_e-c_s))
            # print(rewards)
            #all_time.append(c_e-c_s)

            #c_s = time.time()
            rewards = np.array(rewards)

            index = np.argsort(-rewards) # decendent order
            # select the top percent expressions
            if rewards[index[0]] >= global_opt['reward']:
                if global_opt['complex'] > th.sum(adjs[index[0]]).item():
                    global_opt['reward'] = rewards[index[0]]
                    global_opt['expr'] = infixes[index[0]]
                    global_opt['complex'] = th.sum(adjs[index[0]]).item()
                    global_nodes = nodes[index[0]]
                    global_adjs = adjs[index[0]]
                else:
                    global_opt['reward'] = rewards[index[0]]
                    global_opt['expr'] = infixes[index[0]]
                    global_opt['complex'] = th.sum(adjs[index[0]]).item()
                    global_nodes = nodes[index[0]]
                    global_adjs = adjs[index[0]]
            epoch_opt['reward'] = rewards[index[0]]
            epoch_opt['expr'] = infixes[index[0]]
            epoch_opt['complex'] = th.sum(adjs[index[0]]).item()

            # rewards = np.random.rand(args.batch_size)
            # index = np.argsort(-rewards) # decendent order

            # policy gradients
            threshold = th.tensor(rewards[index[int(args.risk*args.batch_size)]])
            index = index[:int(args.risk*args.batch_size)]
            rewards = th.tensor(rewards)

            # Compute risk seeking and entropy gradient
            risk_seeking_grad_A = th.sum((rewards[index]-threshold).view(-1,1) * logits_A[index], axis=1)
            risk_seeking_grad_X = th.sum((rewards[index]-threshold).view(-1,1) * logits_X[index], axis=1)

            # Mean reduction and clip to limit exploding gradients
            risk_seeking_grad_A = th.clip(th.sum(risk_seeking_grad_A)/len(index), -1e6, 1e6)
            risk_seeking_grad_X = th.clip(th.sum(risk_seeking_grad_X)/len(index), -1e6, 1e6)

            # Compute loss and backpropagate
            optim.zero_grad()
            loss = -1*args.learning_rate*(risk_seeking_grad_A+risk_seeking_grad_X)

            
            print("Global Best:")
            print("  reward:", global_opt['reward'])
            print("  complex:", global_opt['complex'])
            print("Epoch Best:")
            print("  reward:", epoch_opt['reward'])
            print("  complex:", epoch_opt['complex'])
            print("mean constant optimizing time:", np.mean(np.array(all_time)))
            print()

            loss.requires_grad_(True)
            
            #backS = time.time()
            loss.backward()
            optim.step()
            #c_e = time.time()
            #print("backpropagation spends {}s".format(c_e-c_s))
            #backE = time.time()
            #print("backpropagation spends:", backE-backS
            
    # graphs.save_graphs([global_nodes], [global_adjs])
    try:
        r2 = R2(y, y_pred.squeeze().detach().numpy())
    except:
        r2 = 0.0
    global_opt['r2'] = r2
    print("---------------------{}-------------------------".format('Final Result'))
    print("Global Best:")
    print("  expression:", global_opt['expr'])
    print("  reward:", global_opt['reward'])
    print("  r2:", global_opt['r2'])
    print("  complex:", global_opt['complex'])
    print("---------------------------------------------------------------------")
    e = time.time()
    time_cost = e-s
    print("time_cost:", time_cost)
    global_opt['time'] = time_cost
    metric = {'r2': r2, 'reward': global_opt['reward'], "complex": global_opt['complex'], 'time': global_opt['time']}
    best_net.save_as_json(expr, save_file_path, metric=metric)
    return (global_opt, time_cost)

def reward(y_true, y_pred):
    # RMSE
    mse = np.sqrt(np.mean(np.square(y_pred-y_true)))
    val = (1/np.std(y_true))*mse
    val = np.nan_to_num(val, nan=10000)
    val = 1 / (1+val)
    return val

def R2(y_true, y_pred):
    r2_value = r2_score(y_true, y_pred)
    if np.isnan(r2_value):
        r2_value = 0.0
    return r2_value
