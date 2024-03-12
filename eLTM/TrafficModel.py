import pyomo.environ as pyo
from TrafficNet import TN
from pyomo.opt import SolverFactory

tau=60
version=1
timestep=4
net_1=TN(tau,version,timestep)
net_1.readCSV()

model=pyo.AbstractModel()

#模型中link分类
model.RL=pyo.Set(initialize=net_1.RL)
model.SL=pyo.Set(initialize=net_1.SL)
model.GL=pyo.Set(initialize=net_1.GL)
model.CL=pyo.Set(initialize=net_1.CL)
model.AL=pyo.Set(initialize=net_1.AL)
model.L_SL=model.AL-model.SL
model.L_SL_CL=model.L_SL-model.CL
model.L_CL=model.AL-model.CL

#node分类
model.RN=pyo.Set(initialize=net_1.RN)
model.SN=pyo.Set(initialize=net_1.SN)
model.AN=pyo.Set(initialize=net_1.AN)
model.N_RN_SN=model.AN-model.SN-model.RN

#下标集合
model.K=pyo.RangeSet(1,net_1.period-1)
model.AK=pyo.RangeSet(0,net_1.period-1)
model.E=pyo.RangeSet(1,net_1.maxEL)
model.C=pyo.Set(initialize=net_1.e_class)

#设置参数的函数
def alphaRule(model,l,k):
    k=str(k)
    return net_1.alphadf.loc[l,k]
def pathsRule(model,r):
    return net_1.Allpaths[r]
def ODpairRule(model,w):
    return net_1.OD_pair[w]
def ginnodeRule(model,a):
    return net_1.g_in[a]
def goutnodeRule(model,a):
    return net_1.g_out[a]
def einnodeRule(model,a):
    return net_1.e_in[a]
def eoutnodeRule(model,a):
    return net_1.e_out[a]
def DGRule(model,l,a,k):
    if k==0:
        return 0
    k=str(k)
    if (l,a) not in net_1.demandg_df.index:
        return 0
    return net_1.demandg_df.loc[(l,a),k]
def DERule(model,l,a,k,c,e):
    if k==0:
        return 0
    k=str(k)
    if (l,a,c,e) not in net_1.demande_df.index:
        return 0
    return net_1.demande_df.loc[(l,a,c,e),k]
def VRule(model,l):
    return net_1.linkdf.loc[l,'v']
def WRule(model,l):
    return net_1.linkdf.loc[l,'w']
def LRule(model,l):
    return net_1.linkdf.loc[l,'l']
def maxQinRule(model,l,k):
    return net_1.maxQindf.loc[l,'0']
def maxQoutRule(model,l,k):
    return net_1.maxQoutdf.loc[l,'0']
def max_NRule(model,l):
    return net_1.linkdf.loc[l,'max_N']
#参数集合
model.alpha=pyo.Param(model.CL,model.K,initialize=alphaRule)
model.paths=pyo.Param(net_1.Allpaths.keys(),initialize=pathsRule,within=pyo.Any)
model.OD_pair=pyo.Param(net_1.OD_pair.keys(),initialize=ODpairRule,within=pyo.Any)
model.g_in_node=pyo.Param(model.N_RN_SN,initialize=ginnodeRule,within=pyo.Any)
model.g_out_node=pyo.Param(model.N_RN_SN,initialize=goutnodeRule,within=pyo.Any)
model.e_in_node=pyo.Param(model.N_RN_SN,initialize=einnodeRule,within=pyo.Any)
model.e_out_node=pyo.Param(model.N_RN_SN,initialize=eoutnodeRule,within=pyo.Any)
model.DG=pyo.Param(model.RL,model.SN,model.K,initialize=DGRule)
model.DE=pyo.Param(model.RL,model.SN,model.K,model.C,model.E,initialize=DERule)
model.period=pyo.Param(initialize=net_1.period)
model.eMax=pyo.Param(initialize=net_1.maxEL)
model.V=pyo.Param(model.AL,initialize=VRule)
model.W=pyo.Param(model.AL,initialize=WRule)
model.L=pyo.Param(model.AL,initialize=LRule)
model.maxQin=pyo.Param(model.AL,model.K,initialize=maxQinRule)
model.maxQout=pyo.Param(model.AL,model.K,initialize=maxQoutRule)
model.max_N=pyo.Param(model.AL,initialize=max_NRule)

#设置变量
model.UG=pyo.Var(model.L_CL,model.SN,model.AK,within=pyo.NonNegativeReals)
model.VG=pyo.Var(model.L_CL,model.SN,model.AK,within=pyo.NonNegativeReals)
model.UE=pyo.Var(model.AL,model.SN,model.AK,model.C,model.E,within=pyo.NonNegativeReals)
model.VE=pyo.Var(model.AL,model.SN,model.AK,model.C,model.E,within=pyo.NonNegativeReals)
model.x=pyo.Var(model.CL,model.SN,model.AK,model.C,model.E,within=pyo.NonNegativeReals,initialize=0)
model.xd=pyo.Var(model.CL,model.SN,model.AK,model.C,model.E,within=pyo.NonNegativeReals,initialize=0)

#目标函数
def objRule(model):
    x=sum(model.UG[l,a,k]-model.VG[l,a,k] for l in model.L_SL_CL for a in model.SN for k in model.K)
    y=sum(model.UE[l,a,k,c,e]-model.VE[l,a,k,c,e] for l in model.L_SL for a in model.SN for k in model.K for c in model.C for e in model.E)
    return (x+y)
model.obj=pyo.Objective(rule=objRule,sense=pyo.minimize)

###油车约束###
#流量约束1 正向传播
def constr_g_limit_1(model,l,a,k):
    t=k-model.V[l]
    if t<0:
        return pyo.Constraint.Skip
    return model.VG[l,a,k]-model.UG[l,a,t]<=0
model.constr_g_limit_1=pyo.Constraint(model.L_CL,model.SN,model.K,rule=constr_g_limit_1)

#需求满足约束
def constr_g_demand(model,l,a,k):
    return model.UG[l,a,k]==model.DG[l,a,k]
model.constr_g_demand=pyo.Constraint(model.RL,model.SN,model.K,rule=constr_g_demand)

#node守恒约束
def constr_g_conversation(model,i,a,k):
    x=sum(model.UG[l,a,k] for l in model.g_out_node[i])
    y=sum(model.VG[l,a,k] for l in model.g_in_node[i])
    return x-y==0
model.constr_g_conversation=pyo.Constraint(model.N_RN_SN,model.SN,model.K,rule=constr_g_conversation)

#累计流递增
def constr_g_u_increase(model,l,a,k):
    if k==0:
        return model.UG[l,a,0]==0
    return model.UG[l,a,k]-model.UG[l,a,k-1]>=0
model.constr_g_u_increase=pyo.Constraint(model.L_CL,model.SN,model.AK,rule=constr_g_u_increase)

def constr_g_v_increase(model,l,a,k):
    if k==0:
        return model.VG[l,a,0]==0
    return model.VG[l,a,k]-model.VG[l,a,k-1]>=0
model.constr_g_v_increase=pyo.Constraint(model.L_CL,model.SN,model.AK,rule=constr_g_v_increase)

#需求初始化

###电车约束###
#流量约束1 正向传播
def constr_e_limit_1(model,l,a,k,c,e):
    q=e+model.L[l]
    if q>model.eMax:
        return model.VE[l,a,k,c,e]==0
    else:
        t=k-model.V[l]
        if t<0:
            return pyo.Constraint.Skip
        return model.VE[l,a,k,c,e]-model.UE[l,a,t,c,q]<=0
model.constr_e_limit_1=pyo.Constraint(model.L_CL,model.SN,model.K,model.C,model.E,rule=constr_e_limit_1)

#流量约束2 Of
def constr_e_limit_2(model,l,k):
    x=sum(model.VG[l,a,k]-model.VG[l,a,k-1] for a in model.SN)
    y=sum(model.VE[l,a,k,c,e]-model.VE[l,a,k-1,c,e] for a in model.SN for c in model.C for e in model.E)
    return x+y-model.maxQout[l,k]<=0
model.constr_e_limit_2=pyo.Constraint(model.L_CL,model.K,rule=constr_e_limit_2)

#流量约束3 反向传播
def constr_e_limit_3(model,l,k):
    t=k-model.W[l]
    if t<0:
        return pyo.Constraint.Skip
    x=sum(model.UG[l,a,k]-model.VG[l,a,t] for a in model.SN)
    y=sum(model.UE[l,a,k,c,e]-model.VE[l,a,t,c,e] for a in model.SN for c in model.C for e in model.E)
    return x+y-model.max_N[l]<=0
model.constr_e_limit_3=pyo.Constraint(model.L_CL,model.K,rule=constr_e_limit_3)

#流量约束4 If
def constr_e_limit_4(model,l,k):
    x=sum(model.UG[l,a,k]-model.UG[l,a,k-1] for a in model.SN)
    y=sum(model.UE[l,a,k,c,e]-model.UE[l,a,k-1,c,e] for a in model.SN for c in model.C for e in model.E)
    return x+y-model.maxQin[l,k]<=0
model.constr_e_limit_4=pyo.Constraint(model.L_CL,model.K,rule=constr_e_limit_4)

#需求满足约束
def constr_e_demand(model,l,a,k,c,e):
    return model.UE[l,a,k,c,e]-model.DE[l,a,k,c,e]==0
model.constr_e_demand=pyo.Constraint(model.RL,model.SN,model.K,model.C,model.E,rule=constr_e_demand)

#守恒约束
def constr_e_conversation(model,i,a,k,c,e):
    x=sum(model.VE[l,a,k,c,e] for l in model.e_in_node[i])
    y=sum(model.UE[l,a,k,c,e] for l in model.e_out_node[i])
    return x-y==0
model.constr_e_conversation=pyo.Constraint(model.N_RN_SN,model.SN,model.K,model.C,model.E,rule=constr_e_conversation)

#累计流递增
def constr_e_u_increase(model,l,a,k,c,e):
    if k==0:
        return model.UE[l,a,k,c,e]==0
    return model.UE[l,a,k,c,e]-model.UE[l,a,k-1,c,e]>=0
model.constr_e_u_increase=pyo.Constraint(model.AL,model.SN,model.AK,model.C,model.E,rule=constr_e_u_increase)

def constr_e_v_increase(model,l,a,k,c,e):
    if k==0:
        return model.VE[l,a,k,c,e]==0
    return model.VE[l,a,k,c,e]-model.VE[l,a,k-1,c,e]>=0
model.constr_e_v_increase=pyo.Constraint(model.AL,model.SN,model.AK,model.C,model.E,rule=constr_e_v_increase)

#初始化变量

###充电过程约束###
#vechile数量守恒
def constr_charge_conversation(model,l,a,k,c,e):
    if k<2:
        return model.xd[l,a,k,c,e]==0
    else:
        x=model.UE[l,a,k-1,c,e]-model.UE[l,a,k-2,c,e]
        y=model.VE[l,a,k-1,c,e]-model.VE[l,a,k-2,c,e]
        return model.x[l,a,k-1,c,e]+x-y-model.xd[l,a,k,c,e]==0
model.constr_charge_conversation=pyo.Constraint(model.CL,model.SN,model.K,model.C,model.E,rule=constr_charge_conversation)

#满电车辆约束
#中电车辆约束
#低电车辆约束
def constr_charge_update(model,l,a,k,c,e):
    tep=e-model.alpha[l,k]
    if e==model.eMax:        
        return model.x[l,a,k,c,e]-sum(model.xd[l,a,k,c,i] for i in range(tep,e+1))==0
    elif (e>model.alpha[l,k]):
        return model.x[l,a,k,c,e]-model.xd[l,a,k,c,tep]==0
    else:
        return model.x[l,a,k,c,e]==0
model.constr_charge_update=pyo.Constraint(model.CL,model.SN,model.K,model.C,model.E,rule=constr_charge_update)

#流量约束
def constr_charge_limit(model,l,a,k,c,e):
    return model.VE[l,a,k,c,e]-model.VE[l,a,k-1,c,e]-model.x[l,a,k,c,e]<=0
model.constr_charge_limit=pyo.Constraint(model.CL,model.SN,model.K,model.C,model.E,rule=constr_charge_limit)

#容量约束
def constr_charge_cap(model,l,k):
    x=sum(model.UE[l,a,k,c,e]-model.VE[l,a,k,c,e] for a in model.SN for c in model.C for e in model.E)
    return x-model.max_N[l]<=0
model.constr_charge_cap=pyo.Constraint(model.CL,model.K,rule=constr_charge_cap)

#非负性约束
#供应约束
def constr_g_sink(model,l,a,k):
    return model.UG[l,a,k]==0
model.constr_g_sink=pyo.Constraint(model.SL,model.SN,model.K,rule=constr_g_sink)

def constr_e_sink(model,l,a,k,c,e):
    return model.UE[l,a,k,c,e]==0
model.constr_e_sink=pyo.Constraint(model.SL,model.SN,model.K,model.C,model.E,rule=constr_e_sink)


###求解模型###
instance=model.create_instance()
opt=SolverFactory('gurobi')
result=opt.solve(instance,tee=True)
result.write()
instance.solutions.load_from(result)
print(pyo.value(instance.obj))

###导出求解答案###

###测试模型数据###
def test():
    # #测试充电速率参数
    # for i in model.CL:
    #     for j in model.K:
    #         print(model.alpha[i,j])

    #测试需求数据
    # for k in model.K:
    #     for e in model.E:
    #         print(model.DE[0,9,k,1,e])

    #AL数据
    # for i in model.L_SL:
    #     print(i)
    # #xd数据
    # for i in model.CL:
    #     for j in model.SN:
    #         print(pyo.value(model.xd[i,j,0,1,1]))
    for i in model.CL:
        print(i)
    for i in model.SN:
        print(i)
    for i in model.AK:
        print(i)
    for i in model.E:
        print(i)
    for i in model.C:
        print(i)

# test()