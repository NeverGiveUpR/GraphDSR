import sympy as sp
from scipy.optimize import minimize
import numpy as np
from utils import *
import Equations as Eq
import FeynmanEquations as Fq
import time
from sklearn.metrics import r2_score
import argparse
import torch

def get_data(expr_name, seed=2024, datatype='Eq'):
    np.random.seed(seed)
    try:
        if datatype == 'Eq':
            X, y = Eq.func_dict[expr_name](n=100)
        else:
            X, y = Fq.s_func_dict[expr_name](n=100)
    except:
        if datatype == 'Eq':
            X, y = Eq.func_dict[expr_name](num=100)
        else:
            X, y = Fq.s_func_dict[expr_name](num=100)
    X = torch.tensor(X, dtype=float)
    y = torch.tensor(y, dtype=float)
    return X, y

def lbfgs(expr_str, X_train, y_train, X_test, y_test):

    expr = sp.sympify(expr_str[0])

    floats = [f for f in expr.atoms(sp.Float)]
    replacements = {}
    counter = 1

    for f in floats:
        replacement = sp.Symbol(f"c{counter}")
        expr = expr.subs(f, replacement)
        replacements[replacement] = f
        counter += 1


    symbols = sp.symbols('x1:%d' % (input_num+1))

    def loss_function(params):
        temp_replacements = dict(zip(replacements.keys(), params))
        expr_func = sp.lambdify(symbols, expr.subs(temp_replacements), 'numpy')
        y_pred = np.apply_along_axis(lambda row: expr_func(*row), 1, X_train)
        mse = ((y_pred-y_train)**2).mean()
        return mse

    initial_params = []
    for v in replacements.values():
        if abs(float(v))<1e-5:
            initial_params.append(0.0)
        else:
            initial_params.append(float(v))
    # initial_params = [float(v) for v in replacements.values()]
    s = time.time()
    optimization_result = minimize(loss_function, initial_params, method='L-BFGS-B', options={'maxiter': 10000, 'maxfun': 50000})
    print("L-BFGS-B spends {}s".format(time.time()-s))

    if optimization_result.success:
        optimized_params = optimization_result.x
        print("Optimization successful!")
    else:
        print("Optimization failed:", optimization_result.message)
        return

    for (sym, _), val in zip(replacements.items(), optimized_params):
        replacements[sym] = val

    updated_replacements = {str(k): v for k, v in replacements.items()}
    print("Updated replacements:", updated_replacements)

    expr_func_optimized = sp.lambdify(symbols, expr.subs(replacements), 'numpy')
    y_pred_optimized = np.apply_along_axis(lambda row: expr_func_optimized(*row), 1, X_test)
    # y_pred_optimized = expr_func_optimized(X[:,0])
    print("Optimized expression:", expr.subs(replacements))
    print()
    mse_optimized = ((y_pred_optimized - y_test) ** 2).mean()
    print("Optimized MSE (test):", mse_optimized)
    r2_optimized = r2_score(y_test, y_pred_optimized)
    print("Optimized R2 (test):", r2_optimized)
    print()
    y_pred_optimized = np.apply_along_axis(lambda row: expr_func_optimized(*row), 1, X_train)
    mse_optimized = ((y_pred_optimized - y_train) ** 2).mean()
    print("Optimized MSE (train):", mse_optimized)
    r2_optimized = r2_score(y_train, y_pred_optimized)
    print("Optimized R2 (train):", r2_optimized)

    optimed_expr = expr.subs(replacements)
    optimed_expr = sp.simplify(optimed_expr)
    print("optimed_expr:", optimed_expr)
    print("expr_name:", expr_name)

# with open('./config.json', encoding='utf-8') as f:
#     config = json.load(f)
#     config = recursively_convert_to_namespace(config)

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='nguyen1')
parser.add_argument('--dataset', type=str, help='data type', default='Eq')
args = parser.parse_args()
expr_name = args.name
print("expr_name:", expr_name)

X_train, y_train = get_data(expr_name, seed=2024, datatype=args.type)
X_train = X_train.numpy()  
y_train = y_train.numpy()
X_test, y_test = get_data(expr_name, seed=2025, datatype=args.type)
X_test = X_test.numpy()
y_test = y_test.numpy()
print("X_train:{}, y_train:{}".format(X_train.shape, y_train.shape))

input_num = X_train.shape[1]
output_num = 1
print("input_num:", input_num)
print("output_num:", output_num)

print()
print("The first expression:")
expr_str = ['0.685012 - 1.747961*cos(0.004839*x1 - 0.02405*cos(0.000907*x1 + 2.313183) + 1.709374*cos(3.498952*x1 + 0.003828) - 0.043285)']
expr_str = ['0.2661285400390625*sqrt(0.7149161696434021*log(0.5750758647918701*log(1.202698826789856*x2)))']
lbfgs(expr_str, X_train, y_train, X_test, y_test)
