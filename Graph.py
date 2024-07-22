'''
create graph class to represent expressions
'''

import torch as th
import torch.distributions as dist
import torch.nn.functional as F
import random
import copy
import os

from plot import print_graph

class ExprGraph:
    '''
    represent an expression as a undirected acyclic graph
    each node is a element from {sin, cos, x1,...}
    each edge (i, j) is a connection from node i to node j, determined by a one-hot vector (binary)
    an adjcent matrix 
    '''
    def __init__(self, node_num, operators, args, additional_cnst=True):
        self.operators = operators
        self.node_num = node_num # the maximum node number of a graph
        self.var_num = self.operators.var_num
        self.d_node = len(self.operators.library) #  node dimension
        self.d_edge = 2 # edge dimension
        # nested trig function constraint, inverse function constraint
        self.additional_constraint = additional_cnst
        if self.additional_constraint:
            print("Additional constraints are applied.")
        else:
            print("Additional constraints are not applied.")
        self.batch = args.batch_size
        self.args = args
    
    def root_constraint(self, vec):
        # root can not be variables
        # vec: node distributions, batch x d_node, the root node
        vec = F.softmax(vec, dim=1)
        for i in range(self.operators.token2id['x1'], self.d_node-1):
            vec[:, i] = 0
        vec = vec / (th.sum(vec, axis=1)[:, None]+1e-10)
        return vec
    
    def calculate_col_arity(self, nodes, adjs):
        batch, n = nodes.shape
        col_cst = th.sum(adjs[:, :n, :n], dim=1) 
        col_arity = [[self.operators.arity_i(j) for j in nodes[i]] for i in range(batch)]
        col_arity = th.tensor(col_arity)
        return col_cst, col_arity
    
    def minimum_connect_constraint(self, edge_i, P_A):
        # each new node at least has one connection with the graph
        batch = edge_i.shape[0]
        no_cnt = th.sum(edge_i, dim=1)==0
        can_cnt = th.nonzero(P_A)
        no_node = th.sum(P_A, dim=1) != 0

        for i in range(batch):
            temp = []
            for line in can_cnt:
                if line[0] == i:
                    temp.append(line[1])
            if no_cnt[i] and no_node[i]:
                random.shuffle(temp)
                edge_i[i][temp[0]] = 1

        return edge_i
    
    def add_edge_to_adj_undirected(self, adjs, edge_i):
        _, n = edge_i.shape
        adjs[:, n-1, :n] = edge_i
        return adjs
    
    def sample_finish_check(self, nodes, adjs):
        # when all the arity of nonvariable operators statisfied, 
        # a complete graph has been sampled
        # param nodes: node vector, batch x node_num
        # param adjs: adjacent matrix, batch x node_num x node_num
        col_cst, col_arity = self.calculate_col_arity(nodes, adjs)
        masks = col_cst==col_arity
        return masks.all(dim=1)

    def node_constraint(self, vec_, nodes, masks):
        vec = vec_.clone()
        variable_index = self.operators.var_list_index
        index = self.operators.token2id['x1']
        # print("constraint, nodes:", nodes)

        # All variable constraint
        # all the variables should be included in the DAG
        # constrain it with arity, if the arity is less than 

        # maximum node constraint
        # if the current graph has no vairable, only variable can be selected as new node
        # remove the padding node
        valid_nodes = nodes==self.operators.token2id['P']
        valid_nodes = (~valid_nodes).float().sum(dim=1)
        for i in range(len(vec)):
            for j in range(self.var_num, 0, -1):
                # j nodes left to sample
                if valid_nodes[i] == self.node_num - j:
                    t_nodes = list(set(nodes[i].tolist())&set(variable_index))
                    # has no variable
                    if len(t_nodes) == 0:
                        vec[i][:index] = 0.0

        # unique terminal node constraint, each variable can only has one node
        for i in range(len(vec)):
            for index in variable_index:
                if index in nodes[i].tolist():
                    vec[i][index] = 0.0
        
        # not allow to sample Padding nodes
        smp = masks.all(dim=-1)
        for i in range(len(vec)):
            # the complete expression is sampled, use padding
            if smp[i] == False:
                vec[i][:-1] = 0.0
                vec[i][-1] = 1.0
            else:
                # continue sample, padding is not allowed
                vec[i][-1] = 0.0

        vec = vec / (th.sum(vec, axis=1)[:, None]+1e-10)
        spl = th.sum(vec, dim=1) != 0.0

        return vec, spl
    
    def additional_node_constraint(self, nodes, adjs, masks=None):
        # make sure whether trig functions can be the new node
        
        batch, n = nodes.shape
        P_A_masks = th.zeros(batch, n+1)
        col_cst, col_arity = self.calculate_col_arity(nodes, adjs)
        Dicts = []
        for i in range(batch):
            Dict = {}
            has_cnt = col_cst[i] == col_arity[i]
            # print("has_cnt:", has_cnt, "i:", i)
            for j in range(len(has_cnt)):
                if has_cnt[j] == True:
                    t_vec = th.zeros(self.d_node)
                    t_vec[:self.operators.token2id['P']] = 1.0
                else:
                    P_A_masks[i][j] = 1.0
                    t_vec = th.ones(self.d_node)
                    # trig can not be nested
                    adj_ = th.zeros((n+1, n+1))
                    # print("adj_:", adj_.shape, "adjs[i]:", adjs[i].shape)
                    adj_[:-1,:-1] = adjs[i][:n,:n]
                    adj_[-1][j] = 1.0
                    parents = self.depth_first_search_adj(adj_, n)
                    parents = [nodes[i][p-1].item() for p in parents]
                    # print("parents:", parents)
                    # print()
                    if len(set(parents)&set(self.operators.trig_index))!=0:
                        t_vec[self.operators.token2id['sin']] = 0.0
                        t_vec[self.operators.token2id['cos']] = 0.0
                        
                    # inverse function can not be directly connected
                    if nodes[i][j] == self.operators.token2id['sqrt']:
                        t_vec[self.operators.token2id['square']] = 0.0
                    elif nodes[i][j] == self.operators.token2id['square']:
                        t_vec[self.operators.token2id['sqrt']] = 0.0
                    elif nodes[i][j] == self.operators.token2id['log']:
                        t_vec[self.operators.token2id['exp']] = 0.0
                    elif nodes[i][j] == self.operators.token2id['exp']:
                        t_vec[self.operators.token2id['log']] = 0.0
                    else:
                        pass

                    t_vec[self.operators.token2id['P']] = 0.0
                Dict[j] = t_vec
            Dicts.append(Dict)
        
        return P_A_masks, Dicts

    def calculate_priority(self, nodes, adjs):
        prios = []
        node_prios = []
        for adj, node in zip(adjs, nodes):
            cur = [1]
            tmpt = []
            prio = [[1]]
            node_prio = {1:1}
            matrix = (th.nonzero(adj)+1)
            matrix_ = th.zeros(matrix.shape, dtype=int)
            matrix_[:, 0] = matrix[:, 1]
            matrix_[:, 1] = matrix[:, 0]
            matrix_ = matrix_[matrix_[:, 0].sort()[1]]
            matrix_ = matrix_.tolist()

            while len(cur) != 0:
                for line in matrix_:
                    if line[0] in cur:
                        if node[line[1]-1] not in self.operators.var_list_index:
                            tmpt.append(line[1])
                            node_prio[line[1]] = len(prio)+1

                prio.append(tmpt)
                cur = copy.deepcopy(tmpt)
                tmpt = []
            
            node_prios.append(node_prio)
            prios.append(prio) 
      
        return prios, node_prios
    
    def depth_first_search_adj(self, adj, end_node, start_node=0):
        # traverse all the path from root to node_i
        path = []
        stack = [start_node]
        stack_list = []
        stack_list = [th.nonzero(adj[:, stack[-1]]).view(-1).tolist()]

        while stack!=[]:
            neighbors = stack_list[-1]
            if neighbors!=[]:
                ele = neighbors[0]
                stack_list.pop()
                stack_list.append(neighbors[1:])
                stack.append(ele)
                neighbors = th.nonzero(adj[:, ele]).view(-1).tolist()
                temp = list(set(neighbors)-set(stack))
                stack_list.append(temp)
            else:
                stack.pop()
                stack_list.pop()
                continue
            if stack[-1] == end_node:
                path.append([s+1 for s in stack])
                stack.pop()
                stack_list.pop()
        if path!= []:
            parents = []
            for p in path:
                parents.extend(p)
            parents = list(set(parents))
            parents.remove(end_node+1)
            return parents
        else:
            return []
    
    def augment_node_edges(self, nodes, adjs, gnn):
        _, n = nodes.shape
        adjs_ = adjs.clone()
        adjs_[:, n, :n] = 1.0
        # P_A, P_X = self.gnn_update(nodes_, adjs_, gnn)
        P_A, P_X = gnn(nodes, adjs_)
        return P_A, P_X

    def supplement_edges_as_an_expression(self, nodes, adjs, masks):
        batch, n = nodes.shape
        col_cst, col_arity = self.calculate_col_arity(nodes, adjs)
        for i in range(batch):
            finished = col_arity[i] - col_cst[i]
            for j in range(len(finished)):
                if finished[j]>0:
                    parents = self.depth_first_search_adj(adjs[i], j)
                    parents_ind = set([p-1 for p in parents])
                    parents = [nodes[i][p-1].item() for p in parents]
                    # print("i:{},j:{}, finished[j]:{}".format(i, j, finished[j]))
                    # print("parents:", parents)
                    # print("parents_ind:", parents_ind)
                    # could connect
                    index = list(set([k for k in range(n)]) - parents_ind)
                    index.remove(j)
                    
                    index = list(index)
                    # print("index:", index)
                    # add the edges
                    for k in range(int(finished[j])):
                        random.shuffle(index)
                        try:
                            adjs[i][index[0]][j] += 1
                        except:
                            print(adjs[i])
                            print("index:", index)
                            print("j:", j)
                            print("int(finished[j]):", int(finished[j]))
                            self.print_nodes_adjs([nodes[i]], [adjs[i]])
                            quit()

                    # if self.additional_constraint:
                    #     # trig can not be nested
                    #     remain = []
                    #     for k in index:
                    #         adj = adjs[i].clone()
                    #         adj[k][j] = 1.0
                    #         children = self.depth_first_search_adj(adj, None, start_node=j)
                    #         children_ind = set([p-1 for p in children])
                    #         # print("children_ind:", children_ind)
                    #         children = [nodes[i][p-1].item() for p in children]
                    #         # print("children:", children)
                    #         if len(set(children)&set(self.operators.trig_index))==0:
                    #             remain.append(k)
                    #     index = copy.deepcopy(remain)
                    #     # inverse function can not be directly connected
                    #     if len(parents)!=0:
                    #         parent = parents[-1]
                    #         if nodes[i][j] == self.operators.token2id['square'] and \
                    #             parent == self.operators.token2id['sqrt']:
                    #             index = set(index)-set([list(parents_ind)[-1]])
                    #         elif nodes[i][j] == self.operators.token2id['sqrt'] and \
                    #             parent == self.operators.token2id['square']:
                    #             index = set(index)-set([list(parents_ind)[-1]])
                    #         elif nodes[i][j] == self.operators.token2id['exp'] and \
                    #             parent == self.operators.token2id['log']:
                    #             index = set(index)-set([list(parents_ind)[-1]])
                    #         elif nodes[i][j] == self.operators.token2id['log'] and \
                    #             parent == self.operators.token2id['exp']:
                    #             index = set(index)-set([list(parents_ind)[-1]])
                    #         else:
                    #             pass
                    #     if len(children)!=0:
                    #         child = children[-1]
                    #         if nodes[i][j] == self.operators.token2id['sqrt'] and \
                    #             child == self.operators.token2id['square']:
                    #             index = set(index)-set([list(children_ind)[-1]])
                    #         elif nodes[i][j] == self.operators.token2id['square'] and \
                    #             child == self.operators.token2id['sqrt']:
                    #             index = set(index)-set([list(children_ind)[-1]])
                    #         elif nodes[i][j] == self.operators.token2id['exp'] and \
                    #             child == self.operators.token2id['log']:
                    #             index = set(index)-set([list(children_ind)[-1]])
                    #         elif nodes[i][j] == self.operators.token2id['log'] and \
                    #             child == self.operators.token2id['exp']:
                    #             index = set(index)-set([list(children_ind)[-1]])
                    #         else:
                    #             pass
                    
                    

        return nodes, adjs, masks

    def sample_valid_expr_graph_(self, gnn):
        # sample initial node
        vec = self.root_constraint(th.randn(self.batch, self.d_node-1))
        m = dist.Categorical(vec)
        nodes = m.sample().view(-1, 1)
        masks = th.ones((self.batch, 1), dtype=th.bool)
        adjs = th.zeros(self.batch, self.node_num, self.node_num)
        adjs[:, 1, 0] = 1.0

        logits_X = m.log_prob(nodes.view(-1)).view(self.batch, -1)
        entropies_X = m.entropy().view(self.batch, -1)

        logits_A = th.zeros((self.batch, 0))
        # print("logits_A:", logits_A.shape)
        # print("entropies_X:", entropies_X.shape)

        i = 1
        while True:
            i += 1
            # print("^^^^^^^^^^^^^^^^^^^^^^^{}^^^^^^^^^^^^^^^^^^^^^^^".format(i))
            P_A, P_X = self.augment_node_edges(nodes, adjs, gnn)

            # has risk to sample randomly
            prob = th.distributions.Bernoulli(th.tensor(1-self.args.explore_rate)).sample()
            if prob.item() == 0:
                P_X = F.softmax(th.randn(P_X.shape), dim=-1)
                P_A = th.sigmoid(th.randn(P_A.shape))
                
            P_X, spl = self.node_constraint(P_X, nodes, masks)

            if self.additional_constraint:
                P_A_mask, Dicts = self.additional_node_constraint(nodes, adjs, masks)
                P_A = th.multiply(P_A, P_A_mask.float())
                # print("P_A:", P_A)
                # print("P_X:", P_X)
            # print("P_X:", P_X)
            edge_i = th.bernoulli(P_A)
            edge_i = self.minimum_connect_constraint(edge_i, P_A)

            adjs = self.add_edge_to_adj_undirected(adjs, edge_i)
            probs_Ai = th.log(th.clamp(th.sum(th.multiply(P_A, edge_i), dim=1), min=1e-10))
            logits_A = th.cat((logits_A, probs_Ai.view(-1,1)),dim=1)

            # sample node according to the sampled edge_i
            if self.additional_constraint:
                vec = []
                for j in range(len(edge_i)): # batch
                    # select an edge to connect
                    ind = th.nonzero(edge_i[j]).view(-1).tolist()
                    if ind==[]:
                        vec_j = th.zeros(self.d_node)
                        vec_j[self.operators.token2id['P']] = 1.0
                    else:
                        mask = th.ones(self.d_node)
                        for index in ind:
                            mask = th.multiply(Dicts[j][int(index)], mask)
                        if mask[:-1].sum() == 0:
                            vec_j = th.zeros(self.d_node)
                            vec_j[self.operators.token2id['P']] = 1.0
                        else:
                            vec_j = th.multiply(mask, P_X[j])
                        
                    vec.append(vec_j/(th.sum(vec_j)+1e-10))

                vec = th.cat(vec, dim=0).view(self.batch, -1)
                spl = th.sum(vec, dim=1) != 0.0
            else:
                vec = P_X

            node_i_ = th.ones(self.batch).to(th.long)*self.operators.token2id['P']
            m = dist.Categorical(vec)
            node_i = m.sample()
            node_i_ = th.where(spl==True, node_i, node_i_)
            nodes = th.cat([nodes, node_i_[:, None]], dim=1)

            logits_X = th.cat((logits_X, m.log_prob(node_i).view(self.batch, 1)), dim=1)
            entropies_X = th.cat((entropies_X, m.entropy()[:, None]), dim=1)

            new_mask = self.sample_finish_check(nodes, adjs)
            masks = th.cat([masks, th.bitwise_and(~new_mask, masks.all(dim=1))[:,None]], dim=-1)
            # print("masks:", masks)
            if new_mask.all():
                print("all batch are sampled...")
                break
            
            valid_nodes = nodes==self.operators.token2id['P']
            valid_nodes = (~valid_nodes).float().sum(dim=1)
            if th.max(valid_nodes) == self.node_num:
                # print("reached maximum node number...")
                masks[:, -1] = False
                break
        
        nodes, adjs, masks = self.supplement_edges_as_an_expression(nodes, adjs, masks) 

        return nodes, adjs, masks, logits_A, logits_X


    def print_node(self, nodes):
        # node_str = []
        for i in range(len(nodes)):
            temp = []
            for j in range(len(nodes[i])):
                temp.append(self.operators.id2token[int(nodes[i][j])])
            # node_str.append(temp)
            print(i, temp)
        # print("nodes:", node_str)

    def print_adjs(self, adjs):
        for i in range(len(adjs)):
            print("i:", i)
            print(adjs[i])
    
    def print_nodes_adjs(self, nodes, adjs):
        for i in range(len(nodes)):
            temp = []
            for j in range(len(nodes[i])):
                temp.append(self.operators.id2token[int(nodes[i][j])])
            # node_str.append(temp)
            print("i:", i)
            print("     nodes:", temp)
            print("     adjs:", adjs[i])

    def plot_graph(self, node, adj, name):
        node_names = {}
        for i in range(len(node)):
            if node[i] != self.operators.token2id['P']:
                node_names[i+1] = self.operators.id2token[node[i].item()]+'('+str(i+1)+')'
        num = len(node_names.keys())
        print_graph(node_names, adj[:num, :num], name)

    def save_graphs(self, nodes, adjs):
        for i in range(len(nodes)):
            self.plot_graph(nodes[i], adjs[i], 'graph'+str(i))

    def graph_to_infix_(self, root, node, adj):
        # root: the current node index
        # node: the node vector of the current expression
        # adj: the adjacent matrix of the current expression
        
        root_str = self.operators.id2token[node[root]]
        arity = self.operators.arity_i(node[root])

        if arity == 0:
            return root_str
        elif arity == 1:
            index = th.nonzero(adj[:, root]).view(-1).tolist()[0]
            infix = self.graph_to_infix_(index, node, adj)
            return root_str+"("+infix+")"
        elif arity == 2:
            non_zero = th.nonzero(adj[:, root]).view(-1).tolist()
            if len(non_zero) == 2:
                l_index = non_zero[0]
                r_index = non_zero[1]
            else:
                l_index = non_zero[0]
                r_index = non_zero[0]
            l_infix = self.graph_to_infix_(l_index, node, adj)
            r_infix = self.graph_to_infix_(r_index, node, adj)
            return root_str+"("+l_infix+","+r_infix+")"
        else:
            print("Arity error! Exceeding the maximum arity.")
            quit()

    def graph_to_infix(self, nodes, adjs):
        # change an expression graph to infix
        infixes = []
        for node, adj in zip(nodes, adjs):
            # print("--------------------------------------------")
            infix = self.graph_to_infix_(0, node.tolist(), adj)
            infixes.append(infix)

        return infixes
    
    def infix_to_graph(self, infix):
        # change an expression infix to graph
        return
