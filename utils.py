# from sklearn.metrics import r2_score
from scipy.optimize import minimize
import numpy as np
from numpy import sin, cos, tan
import torch as th
import sympy
import copy
import time
import threading
import queue

# rewrite the sqrt function, protected
def sqrt(x):
    return np.sqrt(np.abs(x))

def log(x):
    return np.log(np.abs(x)+1e-5)

def exp(x):
    return np.clip(np.exp(x), 0, 1e5)

def div(x1, x2):
    return x1/(x2+1e-5)

def mul(x1, x2):
    return x1*x2

def sub(x1, x2):
    return x1-x2

def add(x1, x2):
    return x1+x2

def Abs(x):
    return np.abs(x)

def square(x):
    return x**2

def calculate(infix, X):
    y_pred = []
    var_num = X.shape[1]
    for x in X:
        expr = infix
        for i in range(var_num):
            try:
                expr = expr.replace('x'+str(i+1), str(x[i]))
            except:
                pass
        try:
            y_p = eval(expr)
            if y_p < -10000:
                y_p = -10000
            if y_p > 10000:
                y_p = 10000
        except:
            y_p = 10000
        
        y_pred.append(y_p)
    y_pred = np.array(y_pred)
    return y_pred

def t_sin(x):
    return th.sin(x)

def t_cos(x):
    return th.cos(x)

def t_exp(x):
    # print("x:", x)
    # print("t_exp:", th.exp(th.clamp(x, max=1e3)))
    return th.exp(th.clamp(x, max=7))

def t_log(x):
    return th.log(th.abs(x)+1e-5)

def t_sqrt(x):
    return th.sqrt(th.abs(x)+1e-5)

def t_square(x):
    return th.square(x)

def t_tan(x):
    return th.tan(x)

def t_div(x1, x2):
    return x1/(x2+1e-5)

def t_mul(x1, x2):
    return x1*x2

def t_add(x1, x2):
    return x1+x2

def t_sub(x1, x2):
    return x1-x2

fun_dict = {'sin': t_sin, 'cos':t_cos, 'exp':t_exp, 'log': t_log, 
            'sqrt': t_sqrt, 'square': t_square, 'tan': t_tan, 
            'div': t_div, 'mul': t_mul, 'add': t_add, 'sub': t_sub}

def reward(y_true, y_pred):
    # RMSE
    mse = np.sqrt(np.mean(np.square(y_pred-y_true)))
    val = (1/np.std(y_true))*mse
    val = np.nan_to_num(val, nan=10000)
    val = 1 / (1+val)
    return val

def r2(y_obs, y_sim):
    y_obs_mean = np.mean(y_obs)
    y_sim_mean = np.mean(y_sim)
    sum1 = 0
    sum2 = 0
    sum3 = 0
    for i in range(len(y_obs)):
        sum1 = sum1 + (y_sim[i] - y_sim_mean) * (y_obs[i] - y_obs_mean)
        sum2 = sum2 + ((y_sim[i] - y_sim_mean) ** 2)
        sum3 = sum3 + ((y_obs[i] - y_obs_mean) ** 2)
    R2 = (sum1 / ((sum2 ** 0.5) * (sum3 ** 0.5))) ** 2
    return R2

def constant_optimizie_with_const_placeholder(node, adj, operators, X, y):
    def graph_to_infix_str(root):
        
        root_str = operators.id2token[node[root].item()]
        arity = operators.arity_i(node[root].item())

        if arity == 0:
            return str(root_str)
        elif arity == 1:
            index = th.nonzero(adj[:, root]).view(-1).tolist()[0]
            infix = graph_to_infix_str(index)
            if root_str=='square':
                return '{}**2'.format(str(infix))
            elif root_str == 'sqrt':
                return '{}(Abs({}))'.format(root_str, str(infix))
            elif root_str == 'log':
                return '{}(Abs({}))'.format(root_str, str(infix))
            else:
                return '{}({})'.format(root_str, str(infix))
        elif arity == 2:
            non_zero = th.nonzero(adj[:, root]).view(-1).tolist()
            if len(non_zero) == 2:
                l_index = non_zero[0]
                r_index = non_zero[1]
            else:
                l_index = non_zero[0]
                r_index = non_zero[0]
            l_infix = graph_to_infix_str(l_index)
            r_infix = graph_to_infix_str(r_index)

            if root_str == 'add':
                return '{}+{}'.format(str(l_infix), str(r_infix))
            elif root_str == 'sub':
                return '{}-{}'.format(str(l_infix), str(r_infix))
            elif root_str == 'mul':
                return '({})*({})'.format(str(l_infix), str(r_infix))
            else:
                return '({})/({}+1e-5)'.format(str(l_infix), str(r_infix))
        else:
            print("Arity error! Exceeding the maximum arity.")
            quit()


    expr_str = graph_to_infix_str(0)
    print("expr_str:", expr_str)
    sympy_expr = sympy.sympify(expr_str)
    expr = sympy.simplify(sympy_expr)
    print("sympy_form:", expr)
    print()
    quit()
    # expr = BFGS(sympy_expr, X.numpy(), y.numpy())
    return expr

def constant_optimize_an_expr(node, adj, operators, X, y):
    
    parameter = th.randn(int(th.sum(adj).item()+1), requires_grad=True)
    edges = th.nonzero(adj)
    p_dict = {}
    for i in range(len(edges)):
        p_dict[str(edges[i].tolist())] = parameter[i]

    var_num = X.shape[1]
    x1 = X[:, 0]
    var_dict = {'x1': x1}
    if var_num >= 2:
        x2 = X[:, 1]
        var_dict['x2'] = x2
    if var_num >= 3:
        x3 = X[:, 2]
        var_dict['x3'] = x3

    def graph_to_infix_(root):
        
        root_str = operators.id2token[node[root].item()]
        arity = operators.arity_i(node[root].item())
        # print("root:", root, "root_str:", root_str, "arity:", arity)
        # print()
        if arity == 0:
            return var_dict[root_str]
        elif arity == 1:
            index = th.nonzero(adj[:, root]).view(-1).tolist()[0]
            infix = graph_to_infix_(index)
            return p_dict[str([index, root])]*fun_dict[root_str](infix)
        elif arity == 2:
            non_zero = th.nonzero(adj[:, root]).view(-1).tolist()
            if len(non_zero) == 2:
                l_index = non_zero[0]
                r_index = non_zero[1]
            else:
                l_index = non_zero[0]
                r_index = non_zero[0]
            l_infix = graph_to_infix_(l_index)
            r_infix = graph_to_infix_(r_index)
            return fun_dict[root_str](p_dict[str([l_index, root])]*l_infix, 
                                      p_dict[str([r_index, root])]*r_infix)
        else:
            print("Arity error! Exceeding the maximum arity.")
            quit()

    def graph_to_infix_str(root):
        
        root_str = operators.id2token[node[root].item()]
        arity = operators.arity_i(node[root].item())

        if arity == 0:
            return str(root_str)
        elif arity == 1:
            index = th.nonzero(adj[:, root]).view(-1).tolist()[0]
            infix = graph_to_infix_str(index)
            cst = p_dict[str([index, root])].item()
            if root_str=='square':
                return '{}({:.6f}*{})'.format(root_str, cst, str(infix))
            elif root_str == 'sqrt':
                return '{}(Abs({:.6f}*{}))'.format(root_str, cst, str(infix))
            elif root_str == 'log':
                return '{}(Abs({:.6f}*{}))'.format(root_str, cst, str(infix))
            else:
                return '{}({:.6f}*{})'.format(root_str, cst, str(infix))
        elif arity == 2:
            non_zero = th.nonzero(adj[:, root]).view(-1).tolist()
            if len(non_zero) == 2:
                l_index = non_zero[0]
                r_index = non_zero[1]
            else:
                l_index = non_zero[0]
                r_index = non_zero[0]
            l_infix = graph_to_infix_str(l_index)
            r_infix = graph_to_infix_str(r_index)
            l_cst = p_dict[str([l_index, root])].item()
            r_cst = p_dict[str([r_index, root])].item()
            if root_str == 'add':
                return '{:.6f}*{}+{:.6f}*{}'.format(l_cst, str(l_infix), r_cst, str(r_infix))
            elif root_str == 'sub':
                return '{:.6f}*{}-{:.6f}*{}'.format(l_cst, str(l_infix), r_cst, str(r_infix))
            elif root_str == 'mul':
                return '({:.6f}*{})*({:.6f}*{})'.format(l_cst, str(l_infix), r_cst, str(r_infix))
            else:
                return '({:.6f}*{})/({:.6f}*{})'.format(l_cst, str(l_infix), r_cst, str(r_infix))
        else:
            print("Arity error! Exceeding the maximum arity.")
            quit()


    # optim = th.optim.Adam([parameter], lr=0.01)
    # optim = th.optim.Adagrad([parameter], lr=0.01)
    # for i in range(100):
    #     y_pred = parameter[-1]*graph_to_infix_(0)
    #     y_pred = th.clamp(y_pred, min=-1e3, max=1e3)
    #     loss = th.mean(th.square(y_pred-y))
    #     # print("{} | loss:{}".format(i, loss.item()))
    #     optim.zero_grad()
    #     try:
    #         loss.backward()
    #         th.nn.utils.clip_grad_norm_([parameter], max_norm=1, norm_type=2)
    #         optim.step()
    #     except:
    #         pass
    # print("SGD loss:{}".format(loss.item()))
    # re = reward(y.numpy(), y_pred.detach().numpy())
    # print("reward:", re)

    expr = str(parameter[-1].item()) + '*' + graph_to_infix_str(0)
    # print("SGD expr:", expr)
    s = time.time()
    try:
        sympy_expr = stop_after_timeout(sympy_form, 5, expr)
        expr = stop_after_timeout(BFGS, 10, sympy_expr, X.numpy(), y.numpy())
    except:
        # print("exceed sympy")
        sympy_expr = sympy.sympify(expr, dict(square=square))
        expr = sympy_expr

    e = time.time()
    # print(expr)
    # print(e-s, "s")
    # print()
    return str(expr)
  
def sympy_form(expr):
    expr = sympy.sympify(expr, dict(square=square))
    
    try:
        expr = sympy.simplify(expr)
    except:
        expr = expr

    # substitute the re, im with empty
    expr_str = str(expr).replace('re', '')
    expr_str = expr_str.replace('im', '')
    expr = sympy.sympify(expr_str)

    return expr

def stop_after_timeout(func, timeout, *args):
    result_queue = queue.Queue()

    def wrapper_function():
        result = func(*args)
        result_queue.put(result)

    thread = threading.Thread(target=wrapper_function)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        # print("Exceeding the maximum rnning time for an expression...")
        thread.join(timeout=0.1)
        if thread.is_alive():
            raise SystemExit()
        
    while not result_queue.empty():
        return result_queue.get()

def BFGS_const(expr, X, y):
    if isinstance(expr, sympy.Float) or \
        isinstance(expr, sympy.Integer):
        return expr
    
    


def BFGS(expr, X, y):
    # expr: sympy expression
    if isinstance(expr, sympy.Float) or \
        isinstance(expr, sympy.Integer):
        return expr
    is_atomic_number = lambda expr: expr.is_Atom and expr.is_number
    c_dict = {}
    all_c = {}
    all_c_number = []
    i = 0
    for subexpr in sympy.preorder_traversal(expr):
        if is_atomic_number(subexpr):
            if isinstance(subexpr, sympy.Float):
                c_dict[subexpr] = 'c'+str(i)
                all_c['c'+str(i)] = float(str(subexpr))
                all_c_number.append(float(str(subexpr)))
                i += 1
    # print(c_dict)

    new_expr = expr.subs(c_dict)

    dicts_ = {}
    for i in range(X.shape[1]):
        dicts_['x'+str(i+1)] = X[:, i]

    def target(constants):
        dicts = copy.deepcopy(dicts_)
        for i in range(len(constants)):
            dicts['c'+str(i)] = constants[i]
        # print("dicts:", dicts)
        y_pred = sympy.lambdify(dicts.keys(), str(new_expr))(**dicts)
        # print("y_pred:", y_pred)
        return np.mean((y_pred-y)**2)

    results = minimize(target, all_c_number, 
                       method='BFGS',
                       options={'maxiter':100}) 
    constants = results.x
    # loss = target(constants)
    learned_expr = sympy.sympify(new_expr)
    for i in range(len(all_c_number)):
        a = sympy.symbols("c"+str(i))
        learned_expr = learned_expr.subs(a, constants[i].round(4))

    return learned_expr

def constant_optimize(nodes, adjs, operators, X, y):

    X = th.tensor(X)
    y = th.tensor(y)

    cst_expr = []
    
    for node, adj in zip(nodes, adjs):

        expr = constant_optimize_an_expr(node, adj, operators, X, y)
        cst_expr.append(str(expr))

    return cst_expr

def calculate_y(sy_expr, X):
    x1,x2, x3 = sympy.symbols("x1 x2 x3")
    if X.shape[1] == 1:
        y_pred = sympy.lambdify([x1], sy_expr, 'numpy')(X[:,0])
    elif X.shape[1] == 2:
        y_pred = sympy.lambdify([x1,x2], sy_expr, 'numpy')(X[:,0], X[:,1])
    elif X.shape[1] == 3:
        y_pred = sympy.lambdify([x1,x2,x3], sy_expr, 'numpy')(X[:,0], X[:,1], X[:,2])
    else:
        print("Error! exceed maximum input number (2).")
        y_pred = []
        return
    return y_pred
