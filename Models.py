import torch as th
import torch.nn as nn   
import torch.nn.functional as F

class GNN(nn.Module):
    def __init__(self, 
                 pass_steps, 
                 node_num, 
                 d_node, 
                 hidden_size, 
                 K=2, 
                 S=10, 
                 ):
        super(GNN, self).__init__()
        self.pass_steps = pass_steps
        self.node_num = node_num
        self.d_node = d_node
        self.hidden_size = hidden_size
        self.K = K # the number of mixture components
        # when K=1, the distribution degenerates to Bernoulli which assumes
        # the ind,ependence of each potential edge conditioned on the existing graph
        self.S = S # the message passing steps
        self.log_probabilities_X = None
        self.log_probabilities_A = None
        self.num_layer = 1

        self.node_proj = nn.Sequential(
            nn.Linear(self.d_node, self.hidden_size)
        )

        self.MLP_alpha = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.K)
        )
        self.MLP_beta = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.d_node)
        )
        self.MLP_theta = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.K)
        )

        self.f = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )

        self.g = nn.Sequential(
            nn.Linear(self.hidden_size+self.node_num, self.hidden_size+self.node_num),
            nn.ReLU(),
            nn.Linear(self.hidden_size+self.node_num, self.hidden_size),
        )

        # self.gru = nn.GRU(self.hidden_size, self.hidden_size, 1, batch_first=True)
        self.gru = nn.GRUCell(input_size=self.hidden_size, hidden_size=self.hidden_size)

    
    def forward(self, nodes, adjs):
        batch, n = nodes.shape
        # create one-hot feature
        hiddens_0 = F.one_hot(nodes, num_classes=self.d_node)
        hiddens_0 = F.pad(hiddens_0, (0,0,0,1)).view(-1, self.d_node).float()
        hiddens = self.node_proj(hiddens_0).view(-1, self.hidden_size)

        # message passing
        for s in range(self.S):
            hiddens = self.message_passing_once(hiddens, 
                                                adjs[:,:n+1,:n+1],
                                                batch = batch,
                                                n = n)

        # output distribution
        P_A, P_X = self.distributions(hiddens, batch, n)
        return P_A, P_X
    
    def distributions(self, hiddens, batch, n):
        hiddens = hiddens.view(batch, n+1, -1)
        P_X = F.softmax(self.MLP_beta(hiddens[:, -1, :]), dim=-1)

        idx_i = n*th.ones(n+1).long()
        idx_j = th.tensor(range(n+1))
        diff = hiddens[:, idx_i, :] - hiddens[:, idx_j, :]

        log_theta = self.MLP_theta(diff)
        log_alpha = self.MLP_alpha(diff)

        prob = (log_alpha + log_theta).sum(dim=-1)
        P_A = th.sigmoid(prob)

        return P_A, P_X
    
    def message_passing_once(self, hiddens, adjs, batch, n):
        node_feat = [F.one_hot(th.tensor(i), num_classes=self.node_num) for i in range(n+1)]
        node_feat = th.cat(node_feat, dim=0).view(1, -1, self.node_num)
        node_feat = th.tile(node_feat, (batch, 1, 1)).view(-1, self.node_num)
        
        hiddens_hat = th.cat([hiddens, node_feat], dim=1)
        adjs_ = adjs.clone().view(-1, n+1)
        edges = th.nonzero(adjs_)

        diff = hiddens[edges[:,0],:] - hiddens[edges[:,1],:]
        msgs = self.f(diff)
        diff_hat = hiddens_hat[edges[:,0],:] - hiddens_hat[edges[:,1],:]
        alphas = th.sigmoid(self.g(diff_hat))

        att_msgs = th.multiply(alphas, msgs)
        merge_msgs = th.zeros(batch*(n+1), att_msgs.shape[1])
        scatter_idx = edges[:,[0]].expand(-1, att_msgs.shape[1])
        merge_msgs = merge_msgs.scatter_add(0, scatter_idx, att_msgs)

        hiddens = self.gru(merge_msgs, hiddens)

        return hiddens
