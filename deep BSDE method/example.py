'''
An example of code that uses the model. 
You can define any of these components yourself.
'''
from util import ParameterResult, MLPGrad
import parameters.FokkerPlanck.functions_params as funp
import deepBSDE.deepBSDER as dbr
import equations as eq
import torch.nn as nn
import json

# load parameters
path = './parameters/FokkerPlanck/Backward.json'
f = open(path,'r')
params = json.load(f)

dis_params = params['High_dimensional']
equation_params = dis_params['equation_params']
model_params = dis_params['model_params']
train_params = dis_params['train_params']

# Instantiate the PDE
equation = eq.FokkerPlanck(
    parameters=equation_params,
    lamb=funp.lamb_bkwd,
    mu=funp.mu_bkwd,
    g=funp.g_bkwd_high_dim,
    f=funp.f_bkwd_high_dim
)

# Instantiate the basis function
dim = equation_params['dim']
N = model_params['N']
model_params['x'] = [[0 for _ in range(dim)],[0 for _ in range(dim)]]
train_params['x'] = [[0 for _ in range(dim)]]
result = ParameterResult(1,value=0.5)
grad = nn.ModuleList([
    ParameterResult(
        dim=dim,
        value=0.5,
        sd=0.05
    )
]+[
    MLPGrad(
        input_dim=dim,
        output_dim=dim,
        hidden_dim=dim+10,
        layer_num=8,
        batch_norm=False
    ) for _ in range(N-1)
])

# Assemble the model
model = dbr.ExplicitDeepBSDE(
    equation=equation,
    result=result,
    grad=grad,
    model_params=model_params
)

# Train the model
loss_values, result_values = dbr.train(
    model=model,
    train_params=train_params
)