import torch as th
import utils
import numpy as np
import time
import sympy

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

    global_opt = {'expr':'', 'reward':0, 'complex': 100000000}
    epoch_opt = {'expr':'', 'reward': 0, 'complex': 100000000}
    global_nodes = None
    global_adjs = None
    Buffer = {} # {'expr': reward}
    
    s = time.time()
    all_time = []
    for t in range(args.epoch):
        c_s = time.time()
        with th.autograd.set_detect_anomaly(True):
            print("----------------------------{}-{}-----------------------------".format(t, expr))
            # sample expr graphs
            nodes, adjs, masks, logits_A, logits_X = graphs.sample_valid_expr_graph_(gnn)
            masks = th.cat([th.ones((args.batch_size, 1)), masks[:, :-1]],dim=1)
            logits_X = th.multiply(logits_X, masks)
            logits_A = th.multiply(logits_A, masks[:, 1:])

            # obtain expressions
            infixes = graphs.graph_to_infix(nodes, adjs)
            # for infix in infixes:
            #     print(infix)
            # quit()
            # use buffer to reduce training time
            s = time.time()
            rewards = []
            c_infixes = []
            if buffer:
                i = 0
                for infix, node, adj in zip(infixes, nodes, adjs):
                    i += 1
                    bs_time = time.time()
                    if infix in Buffer.keys():
                        be_time = time.time()
                        # print("check the buffer time:", be_time-bs_time)
                        rewards.append(Buffer[infix]['reward'])
                        c_infixes.append(Buffer[infix]['expr'])
                    else:
                        be_time = time.time()
                        cs_time = time.time()
                        c_infix = utils.constant_optimize_an_expr(node, 
                                                                  adj, 
                                                                  operators, 
                                                                  th.tensor(X), 
                                                                  th.tensor(y))
                        
                        ce_time = time.time()
                        # print("buffer:", be_time-bs_time, "constant:", ce_time-cs_time)
                        
                        # print("c_infix:", c_infix)
                        # print("infix:", infix)
                        # print("node:", node)
                        # print("adj:", adj)
                        y_pred = utils.calculate(c_infix, X)
                        r = reward(y, y_pred)

                        # early stop
                        if r>=args.stop_criteria:
                            print("Early stopped!")
                            global_opt['complex'] = th.sum(adj).item()
                            global_opt['reward'] = r
                            global_opt['expr'] = c_infix
                            global_nodes = node
                            global_adjs = adj
                            e = time.time()
                            print("time_cost:", e-s)
                            return (global_opt, e-s)
                        
                        # print("optimize an expr:", e-s)
                        # print(i, "reward (SGD):", r)
                        rewards.append(r)
                        c_infixes.append(c_infix)
                        Buffer[infix] = {'reward':r, 'expr':c_infix} 
                infixes = c_infixes
            else:
                # constant optimize
                infixes = utils.constant_optimize(nodes, adjs, operators, X, y)
                # calculate reward through expr
                for infix in infixes:
                    y_pred = utils.calculate(infix, X)
                    r = reward(y, y_pred)
                    # early stop
                    if r>=args.stop_criteria:
                        print("Early stopped!")
                        global_opt['complex'] = th.sum(adj).item()
                        global_opt['reward'] = r
                        global_opt['expr'] = c_infix
                        e = time.time()
                        # print("time_cost:", e-s)
                        return (global_opt, e-s)
                    rewards.append(r)

            c_e = time.time()
            print("all constant optimizing spends {}s".format(c_e-c_s))
            # print(rewards)
            all_time.append(c_e-c_s)

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
            print("  expression:", global_opt['expr'])
            print("  reward:", global_opt['reward'])
            print("  complex:", global_opt['complex'])
            print("Epoch Best:")
            print("  expression:", epoch_opt['expr'])
            print("  reward:", epoch_opt['reward'])
            print("  complex:", epoch_opt['complex'])
            print("mean constant optimizing time:", np.mean(np.array(all_time)))
            print("global optimal node:", global_nodes)
            print("global optimal adj:", global_adjs)
            print()

            # loss = -1*args.learning_rate*risk_seeking_grad_X
            loss.requires_grad_(True)
            
            backS = time.time()
            loss.backward()
            optim.step()
            backE = time.time()
            print("backpropagation spends:", backE-backS)
            
    # graphs.save_graphs([global_nodes], [global_adjs])

    print("---------------------{}-------------------------".format('Final Result'))
    print("Global Best:")
    print("  expression:", global_opt['expr'])
    print("  reward:", global_opt['reward'])
    print("  complex:", global_opt['complex'])
    print("---------------------------------------------------------------------")
    e = time.time()
    time_cost = e-s
    print("time_cost:", time_cost)
    return (global_opt, time_cost)

def reward(y_true, y_pred):
    # RMSE
    mse = np.sqrt(np.mean(np.square(y_pred-y_true)))
    val = (1/np.std(y_true))*mse
    val = np.nan_to_num(val, nan=10000)
    val = 1 / (1+val)
    return val
