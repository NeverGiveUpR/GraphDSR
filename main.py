import argparse
import numpy as np
import pandas as pd
import json
from operators import Operators
from Graph import ExprGraph
from Models import GNN
# from test import *
import torch as th
from train import train
import Equations as Eq
import FeynmanEquations as Fq
import ReusableEquations as RE
import warnings
warnings.filterwarnings("ignore")


def load_defaults_config():
    """
    Load defaults for training args.
    """
    with open('./config.json', 'r') as f:
        return json.load(f)

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)
    return parser

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")

def main():
    # loading hyperparameters
    defaults = dict()
    defaults.update(load_defaults_config())
    parser = argparse.ArgumentParser(description='training args.')
    parser = add_dict_to_argparser(parser, defaults)
    args = parser.parse_args()

    # exprs = list(Eq.func_dict.keys())[57:60]
    # exprs = list(Fq.s_func_dict.keys())[19:25]
    exprs = list(RE.func_dict.keys())[2:6]
    exprs = ['var2_11']
    results = []
    for expr in exprs:
        a_expr = []
        for i in range(1):
            # X, y = Eq.func_dict[expr]()
            X, y = RE.func_dict[expr](n=100)
            # try:
            #     X, y = Fq.s_func_dict[expr](n=100)
            # except:
            #     X, y = Fq.s_func_dict[expr](num=100)
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
            a_expr.append([expr, 
                           global_opt['reward'],
                           global_opt['expr'],
                           global_opt['complex'],
                           time_cst])
            df = pd.DataFrame(a_expr, columns=['dataset', 'reward', 'expr', 'complex','time'])
            df.to_csv('./results/'+expr+'.csv')
            

if __name__=='__main__':
    main()

    '''
    self.library: ['add', 'div', 'sub', 'mul', 'sin', 'cos', 'sqrt', 'square', 'exp', 'log', 'x1', 'x2', 'x3']
self.token2id: {'add': 0, 'div': 1, 'sub': 2, 'mul': 3, 'sin': 4, 'cos': 5, 
    'sqrt': 6, 'square': 7, 'exp': 8, 'log': 9, 'x1': 10, 'x2': 11, 'x3': 12}
self.id2token: {0: 'add', 1: 'div', 2: 'sub', 3: 'mul', 4: 'sin', 5: 'cos', 
    6: 'sqrt', 7: 'square', 8: 'exp', 9: 'log', 10: 'x1', 11: 'x2', 12: 'x3'}

    '''
