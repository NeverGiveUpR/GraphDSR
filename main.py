import argparse
from operators import Operators
from Graph import ExprGraph
from Models import GNN
from new_train import train
import Equations as Eq
import FeynmanEquations as Fq

import warnings
warnings.filterwarnings("ignore")
from utils import load_defaults_config, add_dict_to_argparser

def main():
    # loading hyperparameters
    defaults = dict()
    defaults.update(load_defaults_config())
    parser = argparse.ArgumentParser(description='training args.')
    parser = add_dict_to_argparser(parser, defaults)
    parser.add_argument('--dataset', type=str, default='RE', help='name of the dataset, \'Eq\' for benchmarks, \'Fq\' for AI Feynman')
    parser.add_argument('--expr', type=str, default='nguyen1', help='name of the expression')
    args = parser.parse_args()
    dataset = args.dataset
    expr = args.expr
    print("dataset:{}, expr:{}".format(dataset, expr))

    
    if dataset == 'Fq':
        try:
            X, y = Fq.s_func_dict[expr](n=100)
        except:
            X, y = Fq.s_func_dict[expr](num=100)
    elif dataset == 'Eq':
        X, y = Eq.func_dict[expr]()
    else:
        raise TypeError("No such dataset!")
    var_num = X.shape[1]
    node_num = args.node_num + var_num
    args.var_number = var_num
    operators = Operators(args.operator_list, args.var_number)
    gnn = GNN(args.pass_steps, node_num, 
            len(operators.library), args.hidden_size)

    graphs = ExprGraph(node_num=node_num, operators=operators, 
                    args=args, additional_cnst=True)
    
    print("_______________________ {}{} ___________________________".format("sampling ", expr))
    global_opt, time_cst = train(graphs, 
                                    gnn, 
                                    operators,
                                    args, X=X, y=y,
                                    expr=expr)
    print("global_opt:", global_opt)
    print("Spends {}s".format(time_cst))
            

if __name__=='__main__':
    main()

    '''
    self.library: ['add', 'div', 'sub', 'mul', 'sin', 'cos', 'sqrt', 'square', 'exp', 'log', 'x1', 'x2', 'x3']
self.token2id: {'add': 0, 'div': 1, 'sub': 2, 'mul': 3, 'sin': 4, 'cos': 5, 
    'sqrt': 6, 'square': 7, 'exp': 8, 'log': 9, 'x1': 10, 'x2': 11, 'x3': 12}
self.id2token: {0: 'add', 1: 'div', 2: 'sub', 3: 'mul', 4: 'sin', 5: 'cos', 
    6: 'sqrt', 7: 'square', 8: 'exp', 9: 'log', 10: 'x1', 11: 'x2', 12: 'x3'}

    '''
