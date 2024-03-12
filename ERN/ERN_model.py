import pyomo.environ as pyo
import pandas as pd
from pyomo.opt import SolverFactory
import numpy as np

# #创建csv文件  此部分代码运行一次即可  需要预先创建一个“data”的文件夹
# cells={'id':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
#        'ctype':['CR','CMD','CMD','CO','CMD','CMD','CMD','CS','CO','CMD','CMD','CO','CQ','CC','CQ','CQ','CC','CQ','CQ','CC','CQ'],
#        'maxcap':[-1,1000,1000,1000,1000,1000,1000,-1,1000,1000,1000,1000,20,2,20,20,1,20,20,1,20],
#        'h':[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
#        }
# celldata=pd.DataFrame(cells)
# celldata.to_csv('data\celldata.csv',index=None)

# links={'id':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24],
#        'start':[13,14,19,20,16,17,2,15,10,21,6,18,1,7,2,2,3,4,5,6,9,10,11,12],
#        'end':[14,15,20,21,17,18,13,3,19,11,16,7,2,8,3,9,4,5,6,7,10,11,12,5],
#         'ltype':['C','C','C','C','C','C','D','M','D','M','D','M','O','O','D','D','O','M','O','M','O','D','O','M']
#        }
# linkdata=pd.DataFrame(links)
# linkdata.to_csv('data\linkdata.csv',index=None)


# maxcap={'id':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
#         0:[-1,1000,1000,1000,1000,1000,1000,-1,1000,1000,1000,1000,20,2,20,20,1,20,20,1,20]
#         }
# maxcap_N=pd.DataFrame(maxcap)
# maxcap_N.to_csv('data\maxcap.csv',index=None)

# q={'id':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],
#       0:[-1,200,200,200,200,200,200,-1,200,200,200,200,200,200,200,200,200,200,200,200,200]
#       }
# qdata=pd.DataFrame(q)
# qdata.to_csv('data\qdata.csv',index=None)

# d={'OD_id':[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
#    'EL':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
#    0:[0,0,2,2,2,1,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23,0,0]
#    }
# ddata=pd.DataFrame(d)
# ddata.to_csv('data\ddata.csv',index=None)

# paths={'id':[1,2,3,4,5,6,7],
#        'OD_id':[1,1,1,1,1,1,1],
#        'path':[(1,2,3,4,5,6,7,8),
#                (1,2,13,14,15,3,4,5,6,7,8),
#                (1,2,3,4,5,6,16,17,18,7,8),
#                (1,2,9,10,19,20,21,11,12,5,6,7,8),
#                (1,2,13,14,15,3,4,5,6,16,17,18,7,8),
#                (1,2,9,10,19,20,21,11,12,5,6,16,17,18,7,8),
#                (1,2,9,10,11,12,5,6,7,8)]
#        }
# pathdata=pd.DataFrame(paths)
# pathdata.to_csv('data\pathdata.csv',index=None)

# alpha={'id':[14,17,20],
#        '0':[4,4,4]
#        }
# alphadata=pd.DataFrame(alpha)
# alphadata.to_csv('data//alphadata.csv',index=None)

#读取数据
cell_data=pd.read_csv('data\celldata.csv')
cell_data.set_index('id',inplace=True)

link_data=pd.read_csv('data\linkdata.csv')
link_data.set_index('id',inplace=True)

max_cap=pd.read_csv('data\maxcap.csv')
max_cap.set_index('id',inplace=True)

q_data=pd.read_csv('data\qdata.csv')
q_data.set_index('id',inplace=True)

d_data=pd.read_csv('data\ddata.csv')
d_data.set_index('EL',inplace=True)
# print (d_data.loc[2].index)

path_data=pd.read_csv('data\pathdata.csv')
path_data.set_index('id',inplace=True)

alpha_data=pd.read_csv('data//alphadata.csv')
alpha_data.set_index('id',inplace=True)

#创建模型
# model=pyo.ConcreteModel()
model=pyo.AbstractModel()

#参数区域
cellIndex=cell_data.index.tolist()
linkIndex=link_data.index.tolist()
pathIndex=path_data.index.tolist()
eIndex=d_data.index.tolist()
cellIndex_exCS=cellIndex[:-1]
Td=1
Tf=5
ODindex=[1]
Lmax=len(eIndex)
backParam=1


#创建角标
model.cIndex=pyo.Set(initialize=cellIndex)
model.tIndex=pyo.Set(initialize=[i for i in range(Tf)])
model.pIndex=pyo.Set(initialize=pathIndex)
model.lIndex=pyo.Set(initialize=eIndex)
model.td=pyo.Set(initialize=[i for i in range(Td)])
model.t0=pyo.Set(initialize=[i for i in range(1,Tf)])
model.td1=pyo.Set(initialize=[i for i in range(1,Td+1)])
model.tfextd1=pyo.Set(initialize=[i for i in range(Td+1,Tf)])
model.wIndex=pyo.Set(initialize=ODindex)

#细胞分类
def C_init(cname):
    C=[]
    for i in cellIndex:
        if cell_data.loc[i,'ctype']==cname:
            C.append(i)
    # print(C)
    return C

model.CR=pyo.Set(initialize=C_init('CR'))
model.CS=pyo.Set(initialize=C_init('CS'))
model.CO=pyo.Set(initialize=C_init('CO'))
model.CMD=pyo.Set(initialize=C_init('CMD'))
model.CQ=pyo.Set(initialize=C_init('CQ'))
model.CC=pyo.Set(initialize=C_init('CC'))
CexCS=C_init('CR')+C_init('CO')+C_init('CMD')+C_init('CQ')+C_init('CC')
C_G=C_init('CS')+C_init('CMD')+C_init('CO')
model.CexCS=pyo.Set(initialize=CexCS)
model.C_G=pyo.Set(initialize=C_G)

#链接分类
def L_init(lname):
    L=[]
    for i in linkIndex:
        if link_data.loc[i,'ltype']==lname:
            L.append((link_data.loc[i,'start'],link_data.loc[i,'end']))
    # print(L)
    return L

model.OL=pyo.Set(initialize=L_init('O'))
model.ML=pyo.Set(initialize=L_init('M'))
model.DL=pyo.Set(initialize=L_init('D'))
model.CL=pyo.Set(initialize=L_init('C'))
AL=L_init('O')+L_init('M')+L_init('D')+L_init('C')
model.AL=pyo.Set(initialize=AL)

#路径具体信息
# def P_init():
#     P={}
#     for i in pathIndex:
#         path_detail=path_data.loc[i,'path']
#         pos=0
#         paths=[]
#         while pos!=-1:
#             ptr=path_detail.find(',',pos+1,len(path_detail))
#             paths.append(int(path_detail[pos+1:ptr]))
#             pos=ptr
#         P.update({i:paths})
#     print(P)
#     return P

#路径信息
def P_init(model,w):
    teplist=[]
    for i in pathIndex:
        if path_data.loc[i,'OD_id']==w:
            teplist.append(i)
    return teplist
model.pdata=pyo.Param(model.wIndex,initialize=P_init,within=pyo.Any)
# model.wIndex.construct()
# model.pdata.construct()
# print(model.pdata[1][2])

#相邻节点信息
def suc_init(model,c):
    teplist=[]
    for i,j in model.AL:
        if i==c:
            teplist.append(j)
    # print(teplist)
    return teplist
model.sucCell=pyo.Param(model.cIndex,initialize=suc_init,within=pyo.Any)
# model.AL.construct()
# model.cIndex.construct()
# model.sucCell.construct()
# for i in model.cIndex:
#     print(model.sucCell[i])

def pre_init(model,c):
    teplist=[]
    for i,j in model.AL:
        if j==c:
            teplist.append(i)
    # print(teplist)
    return teplist
model.preCell=pyo.Param(model.cIndex,initialize=pre_init,within=pyo.Any)

#需求分类
def D_init(model,w,t):
    tepdic={}
    for i in eIndex:
        if d_data.loc[i,'OD_id']==w:
            tepdic.update({i:d_data.loc[i,str(t)]})
    # print(tepdic)
    return tepdic
model.ddata=pyo.Param(model.wIndex,model.td,initialize=D_init,within=pyo.Any)
# model.wIndex.construct()
# model.td.construct()
# model.ddata.construct()
# print(model.ddata[1,0][7])

#充电速率
def a_init(model,i):
    return alpha_data.loc[i,'0']
model.alpha=pyo.Param(model.CC,initialize=a_init)

#最大流量
def Q_init(model,i):
    if q_data.loc[i,'0']==-1:
        return 10000
    else:
        return q_data.loc[i,'0']
model.Q=pyo.Param(model.cIndex,initialize=Q_init)

#最大容量
def N_init(model,i):
    if max_cap.loc[i,'0']==-1:
        return 10000
    else:
        return max_cap.loc[i,'0']
model.N=pyo.Param(model.cIndex,initialize=N_init)

#定义变量
model.x=pyo.Var(model.cIndex,model.tIndex,model.pIndex,model.lIndex,within=pyo.NonNegativeReals,initialize=0)
model.y=pyo.Var(model.AL,model.tIndex,model.pIndex,model.lIndex,within=pyo.NonNegativeReals,initialize=0)
model.d=pyo.Var(model.td,model.pIndex,model.lIndex,within=pyo.NonNegativeReals,initialize=0)
model.xd=pyo.Var(model.CC,model.tIndex,model.pIndex,model.lIndex,within=pyo.NonNegativeReals,initialize=0)

#定义目标函数
def objRule(model):
    return sum(model.x[c,t,r,l] for c in model.CexCS for t in model.tIndex for r in model.pIndex for l in model.lIndex)
model.obj=pyo.Objective(rule=objRule,sense=pyo.minimize)

##定义约束
#起始点需求聚合约束
def constr6Rule(model,w,t,l):
    return sum(model.d[t,r,l] for r in model.pIndex if r in model.pdata[w])==model.ddata[w,t][l]
model.constr6=pyo.Constraint(model.wIndex,model.td,model.lIndex,rule=constr6Rule)

#起始节点流量约束
def constr7aRule(model,i,t ,r,l):
    j=model.sucCell[i][0]
    return model.x[i,t-1,r,l]+model.d[t-1,r,l]-model.y[i,j,t-1,r,l]==model.x[i,t,r,l]
model.constr7a=pyo.Constraint(model.CR,model.td1,model.pIndex,model.lIndex,rule=constr7aRule)

def constr7bRule(model,i,t ,r,l):
    j=model.sucCell[i][0]
    return model.x[i,t-1,r,l]-model.y[i,j,t-1,r,l]==model.x[i,t,r,l]
model.constr7b=pyo.Constraint(model.CR,model.tfextd1,model.pIndex,model.lIndex,rule=constr7bRule)

#一般节点守恒约束
def constr8Rule(model,i,t,r,l):
    if l==Lmax:
        return model.x[i,t-1,r,Lmax]-sum(model.y[i,j,t-1,r,Lmax] for j in model.sucCell[i])==model.x[i,t,r,Lmax]
    else:
        return model.x[i,t-1,r,l]-sum(model.y[i,j,t-1,r,l] for j in model.sucCell[i])+sum(model.y[k,i,t-1,r,l+1] for k in model.preCell[i])==model.x[i,t,r,l]
model.constr8=pyo.Constraint(model.C_G,model.t0,model.pIndex,model.lIndex,rule=constr8Rule)

#排队节点守恒约束
def constr9Rule(model,i,t,r,l):
    j=model.sucCell[i][0]
    k=model.preCell[i][0]
    return model.x[i,t-1,r,l]-model.y[i,j,t-1,r,l]+model.y[k,i,t-1,r,l]==model.x[i,t,r,l]
model.constr9=pyo.Constraint(model.CQ,model.t0,model.pIndex,model.lIndex,rule=constr9Rule)

def constr10Rule(model,i,t,r,l):
    j=model.sucCell[i][0]
    k=model.preCell[i][0]
    return  model.x[i,t-1,r,l]-model.y[i,j,t-1,r,l]+model.y[k,i,t-1,r,l]==model.xd[i,t,r,l]
model.constr10=pyo.Constraint(model.CC,model.t0,model.pIndex,model.lIndex,rule=constr10Rule)

#充电节点更新规则
def constr11Rule(model,i,t,r,l):
    bond=model.alpha[i]
    if l==Lmax:
        return model.x[i,t,r,Lmax]==sum(model.xd[i,t,r,j] for j in range(Lmax-bond,Lmax+1))
    elif l<=bond:
        return model.x[i,t,r,l]==0
    elif l<Lmax:
        return model.x[i,t,r,l]==model.xd[i,t,r,l-bond]
model.constr11=pyo.Constraint(model.CC,model.t0,model.pIndex,model.lIndex,rule=constr11Rule)

#流量大小约束
def constr12aRule(model,i,t,r,l):
    return sum(model.y[i,j,t,r,l] for j in model.sucCell[i])-model.x[i,t,r,l]<=0
model.constr12a=pyo.Constraint(model.cIndex,model.t0,model.pIndex,model.lIndex,rule=constr12aRule)

def constr12bRule(model,i,k,t):
    return sum(model.y[i,j,t,r,l] for j in model.sucCell[i] for r in model.pIndex for l in model.lIndex)<=model.Q[i]
model.constr12b=pyo.Constraint(model.AL,model.t0,rule=constr12bRule)

# CexCR=C_init('CS')+C_init('CO')+C_init('CMD')+C_init('CQ')+C_init('CC')
# model.CexCR=pyo.Set(initialize=CexCS)
def constr12cRule(model,i,j,t):
   return sum(model.y[k,j,t,r,l] for k in model.preCell[j] for r in model.pIndex for l in model.lIndex)<=model.Q[j]
model.constr12c=pyo.Constraint(model.AL,model.t0,rule=constr12cRule)

def constr12dRule(model,j,t):
    return sum(model.y[i,j,t,r,l] for i in model.preCell[j] for r in model.pIndex for l in model.lIndex)+backParam*sum(model.x[j,t,r,l] for r in model.pIndex for l in model.lIndex)<=backParam*model.N[j]
model.constr12d=pyo.Constraint(model.C_G,model.t0,rule=constr12dRule)

def constr12eRule(model,j,t):
    return sum(model.y[i,j,t,r,l] for i in model.preCell[j] for r in model.pIndex for l in model.lIndex)+sum(model.x[j,t,r,l] for r in model.pIndex for l in model.lIndex)<=model.N[j]
model.constr12e=pyo.Constraint(model.CQ,model.t0,rule=constr12eRule)

def constr12fRule(model,j,t):
    return sum(model.y[i,j,t,r,l] for i in model.preCell[j] for r in model.pIndex for l in model.lIndex)-sum(model.y[j,m,t,r,l] for m in model.sucCell[j] for r in model.pIndex for l in model.lIndex)+sum(model.x[j,t,r,l] for r in model.pIndex for l in model.lIndex)<=model.N[j]
model.constr12f=pyo.Constraint(model.CC,model.t0,rule=constr12fRule)

#初始化参数
def constr13Rule(model,i,r,l):
    return model.x[i,0,r,l]==0
model.constr13=pyo.Constraint(model.cIndex,model.pIndex,model.lIndex,rule=constr13Rule)

def constr14Rule(model,i,j,t,r):
    return model.y[i,j,t,r,1]==0
model.constr14=pyo.Constraint(model.AL,model.t0,model.pIndex,rule=constr14Rule)

def constr15Rule(model,i,t,r,l):
    j=model.sucCell[i][0]
    a=str(i)+','+str(j)
    if a not in path_data.loc[i,'path']:
        return model.y[i,j,t,r,l]==0
    else:
        return pyo.Constraint.Skip
model.constr15=pyo.Constraint(model.CR,model.t0,model.pIndex,model.lIndex,rule=constr15Rule)

def constr16Rule(model,i,t,r,l):
    return model.x[i,t,r,l]>=0
model.constr16=pyo.Constraint(model.cIndex,model.tIndex,model.pIndex,model.lIndex)

def constr17Rule(model,i,j,t,r,l):
    return model.y[i,j,t,r,l]>=0
model.constr17=pyo.Constraint(model.AL,model.tIndex,model.pIndex,model.lIndex)

instance=model.create_instance()
opt=SolverFactory('gurobi')
opt.options["LPMethod"] = 4
result=opt.solve(instance,tee=True)
result.write()
instance.solutions.load_from(result)

x_result=np.array([sum(pyo.value(instance.x[c,t,p,l]) for p in instance.pIndex for l in instance.lIndex) for c in instance.cIndex for t in instance.tIndex]).reshape(len(instance.cIndex),len(instance.tIndex))
x_df=pd.DataFrame(x_result,index=instance.cIndex,columns=instance.tIndex)
x_df.to_csv('ans//x_result.csv')

y_result=np.array([sum(pyo.value(instance.y[i,j,t,p,l]) for p in instance.pIndex for l in instance.lIndex) for i,j in instance.AL for t in instance.tIndex]).reshape(len(instance.AL),len(instance.tIndex))
y_df=pd.DataFrame(y_result,index=instance.AL,columns=instance.tIndex)
y_df.to_csv('ans//y_result.csv')

xd_result=np.array([sum(pyo.value(instance.xd[c,t,p,l]) for p in instance.pIndex for l in instance.lIndex) for c in instance.CC for t in instance.tIndex]).reshape(len(instance.CC),len(instance.tIndex))
xd_df=pd.DataFrame(xd_result,index=instance.CC,columns=instance.tIndex)
xd_df.to_csv('ans//xd_result.csv')
