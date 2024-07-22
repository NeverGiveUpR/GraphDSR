import torch as th
import torch.nn.functional as F
import Equations as Eq
import FeynmanEquations as Fq
import ReusableEquations as RE
import utils
import numpy as np

from sympy import symbols, sin, cos  
import sympy
import re
import numpy as np
from utils import BFGS
from scipy.optimize import minimize
import time
from matplotlib import pyplot as plt

# extrapolation ability
# nguyen1_c
dicts = {'Dsr': '(((cos(x1 + x1) - x1) + -1.8056) * x1) * -2.1799',
         'GP-GOMEA': 'x1*(2.249312*x1**2 + 1.874192*exp(x1))',
         'SBP-GP': '2.13357001736726*x1 - (2.075406*x1 + 0.983763626930664)*cos(2.086298*x1) + exp(x1) + \
    sin(x1*sin((sin(x1) + sin(x1 + (sqrt(x1) + 0.793717)*(sqrt(x1) + 0.795336)*log(sin(cos(x1**(3/2))))\
          - 4.659039))*log(x1)*sin(x1) + 0.818831759831205))',
          'Symformer': '(3.7326598167419434*((0.40700066089630127^sin((2.114156723022461+(x1+sin(x1)))))*x1))',
          'NeSymRes': '2.20389963312472*x1/cos(0.691263822699361*x1 + 0.323706336084067)**2',
          'GraphDSR': '3.39*x1**3 + 2.12*x1**2 + 1.78*x1'}

def square(x):
    return x**2

num = 600
x_1 = []
x_2 = []
for i in range(1,num + 1):
    a = np.random.uniform(-3,3)
    x_1.append(a)

x_1 = np.array(x_1)
x_1.sort()
X = x_1[:, np.newaxis]

y = 3.39*x_1**3 + 2.12*x_1**2 + 1.78*x_1

expr = dicts['Dsr']
expr = sympy.sympify(expr, dict(square=square))
y_dsr = utils.calculate_y(expr, X)

expr = dicts['GP-GOMEA']
expr = sympy.sympify(expr, dict(square=square))
y_gp_gomea = utils.calculate_y(expr, X)

expr = dicts['SBP-GP']
expr = sympy.sympify(expr, dict(square=square))
y_sbp_gp = utils.calculate_y(expr, X)

expr = dicts['Symformer']
expr = sympy.sympify(expr, dict(square=square))
y_symformer = utils.calculate_y(expr, X)

expr = dicts['NeSymRes']
expr = sympy.sympify(expr, dict(square=square))
y_nesymres = utils.calculate_y(expr, X)

expr = dicts['GraphDSR']
expr = sympy.sympify(expr, dict(square=square))
y_graphdsr = utils.calculate_y(expr, X)

x = x_1[np.where(np.abs(x_1) <1)]
x = x.reshape(-1,1)
plt.fill_between(x.reshape(1,-1)[0], -30000, 30000, facecolor='g', alpha=0.3)

plt.plot(x_1,y_dsr,c = '#87CEFA',linestyle ='-',label = 'DSR',linewidth=5,markersize = 16)
plt.plot(x_1,y_gp_gomea,c = '#4169E1',linestyle ='-',label = 'GP-GOMEA',linewidth=5,markersize = 16)
plt.plot(x_1,y_sbp_gp,c = 'deepskyblue',linestyle ='-',label = 'SBP-GP',linewidth=5,markersize = 16)
plt.plot(x_1,y_symformer,c = 'cadetblue',linestyle ='-',label = 'Symformer',linewidth=5,markersize = 16)
plt.plot(x_1,y_nesymres,c = 'steelblue',linestyle ='-',label = 'NeSymRes',linewidth=5,markersize = 16)
plt.plot(x_1,y_graphdsr,c = 'orangered',linestyle ='-',label = 'GraphDSR',linewidth=5,markersize = 16)
plt.plot(x_1,y,c = '#696969',linestyle ='--',label = 'True Value',linewidth=2.5,markersize = 16)

plt.xlim(-3,3)
plt.ylim(-10,10)
plt.xlabel('X',size = 15)
plt.ylabel('Y',size = 15)
# plt.legend()
plt.legend(loc='lower right', prop={'size': 10})

# plt.savefig("range1.pdf")
plt.savefig("expend-AAAI-8.png")
# plt.show()
