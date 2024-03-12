# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 13:09:15 2022

@author: zhang
"""

from traffic_network import TN
import pyomo.environ as pyo
import pandas
from pyomo.opt import SolverFactory

model=pyo.AbstractModel()


########
#构建交通网络
########
tau=60
Data_version='1'
Total_timestep=20 #先设个20吧，我也不知道该是几来着
Para=TN(tau,Data_version,Total_timestep)
#由于TN中均是参数这里命名为para


##########
#一大坨集合 V2更新，与老师变量名统一，这样看就不会特别麻烦了
##########
model.K=pyo.RangeSet(0,Para.step_num-1)
model.E=pyo.RangeSet(1,Para.max_EL)
model.C=pyo.Set(initialize=([1])) #这个在类里面没写，等会补一下

model.AR=pyo.Set(initialize=(Para.RL))
model.AS=pyo.Set(initialize=(Para.SL))
model.AG=pyo.Set(initialize=(Para.GL))
model.AC=pyo.Set(initialize=(Para.CL))
model.A=pyo.Set(initialize=(Para.AL))
model.A_AS=model.A-model.AS
model.A_AC=model.A-model.AC
model.A_AS_AC=model.A-model.AS-model.AC

model.NS=pyo.Set(initialize=(Para.SN))
model.NR=pyo.Set(initialize=(Para.RN))
model.N=pyo.Set(initialize=(Para.AN))

##########
#一大坨变量~~~~
##########
model.u_g=pyo.Var(model.A_AC,model.NS,model.K,domain=pyo.NonNegativeReals)
model.v_g=pyo.Var(model.A_AC,model.NS,model.K,domain=pyo.NonNegativeReals)

model.u_e=pyo.Var(model.A,model.NS,model.C,model.E,model.K,domain=pyo.NonNegativeReals)
model.v_e=pyo.Var(model.A,model.NS,model.C,model.E,model.K,domain=pyo.NonNegativeReals)
model.x=pyo.Var(model.AC,model.NS,model.C,model.E,model.K,domain=pyo.NonNegativeReals)
model.dx=pyo.Var(model.AC,model.NS,model.C,model.E,model.K,domain=pyo.NonNegativeReals)

#######
#参数
#######
#Para类
model.chi=pyo.Param(initialize=(1))
#此外无多余参数

###############
#目标函数
###############v
def objrule(model):
    GV=sum(model.u_g[a,s,k]-model.v_g[a,s,k] for a in model.A_AS_AC for k in model.K for s in model.NS)*model.chi
    EV=sum(model.u_e[a,s,c,e,k]-model.v_e[a,s,c,e,k] for c in model.C for e in model.E for k in model.K for s in model.NS for a in model.A_AS)*model.chi
    return (GV + EV)
    
model.obj=pyo.Objective(rule=objrule,sense=pyo.minimize)

##########
#对燃油车的约束
##########

#流量限制
def cons8_rule(model,a,s,k):
    if k == 0:
        return pyo.Constraint.Skip
    t=k-Para.V(a)
    if t < 0:
        return pyo.Constraint.Skip
    else:
        return model.u_g[a,s,k] <= model.v_g[a,s,t]
model.cons8=pyo.Constraint(model.A_AC,model.NS,model.K,rule=cons8_rule)

#需求满足(其实是对于R的守恒定律)
def cons9_rule(model,a,s,k):
    return model.u_g[a,s,k] == Para.Demand_g(a, s, k)
model.cons9=pyo.Constraint(model.AR,model.NS,model.K,rule=cons9_rule)

#守恒定律
def cons10a_rule(model,i,s,k):
    return sum(model.v_g[a, s, k] for a in Para.GV_dict_in[i]) == sum(model.u_g[b, s, k] for b in Para.GV_dict_out[i])
model.cons10a=pyo.Constraint(model.N-model.NR,model.NS,model.K,rule=cons10a_rule)

def cons_sink_link_node_g_rule(model, a, s, k):
    sink_links = Para.GV_dict_in[s]
    if sink_links[0] != a:
        return model.u_g[a, s, k] == 0
    else:
        return pyo.Constraint.Skip
model.cons_sink_link_node_g=pyo.Constraint(model.AS, model.NS, model.K, rule=cons_sink_link_node_g_rule)

#积累车辆U，V不递减
def cons11_rule(model,a,s,k):
    if k == 0:
        return model.v_g[a,s,k] == 0
    else:
        return model.v_g[a,s,k] >= model.v_g[a,s,k-1]
model.con11_13a=pyo.Constraint(model.A_AC,model.NS,model.K,rule=cons11_rule)

def cons12_rule(model,a,s,k):
    if k == 0:
        return model.u_g[a,s,k] == 0
    else:
        return model.u_g[a,s,k] >= model.u_g[a,s,k-1]
model.con12_13b=pyo.Constraint(model.A_AC,model.NS,model.K,rule=cons12_rule)

###############
#EV约束
###############

#流限制
def con16_rule(model,a,s,c,e,k):
    if (k == 0):
        return pyo.Constraint.Skip
    else:
        t=k-Para.V(a)
        if t < 0:
            return pyo.Constraint.Skip
        else:
            if e <= Para.max_EL-Para.L(a):
                return model.v_e[a,s,c,e,k] <= model.u_e[a,s,c,e+Para.L(a),t]
            else:
                return model.v_e[a,s,c,e,k] == 0
model.con16=pyo.Constraint(model.A_AC,model.NS,model.C,model.E,model.K,rule=con16_rule)

#道路Out容量限制
def con17_rule(model,a,k):
    if k == 0:
        return pyo.Constraint.Skip
    else:
        sum_gv=sum(model.v_g[a,s,k]-model.v_g[a,s,k-1] for s in model.NS)
        sum_ev=sum(model.v_e[a,s,c,e,k]-model.v_e[a,s,c,e,k-1] for s in model.NS for c in model.C for e in model.E)
        return sum_gv+sum_ev<=Para.Q_out(a, k)
#model.con17=pyo.Constraint(model.A_AC,model.K,rule=con17_rule)

#道路堵塞密度限制
def con18_rule(model,a,k):
    if k==0:
        return pyo.Constraint.Skip
    else:
        t=k-Para.W(a)
        if t<0:
            return pyo.Constraint.Skip
        else:
            sum_gu_min_v=sum(model.u_g[a,s,k]-model.v_g[a,s,t] for s in model.NS)
            sum_eu_min_v=sum(model.u_e[a,s,c,e,k]-model.v_e[a,s,c,e,t] for s in model.NS for c in model.C for e in model.E)
            return sum_gu_min_v+sum_eu_min_v <= Para.N(a)
model.con18=pyo.Constraint(model.A_AC,model.K,rule=con18_rule)   
        
#道路In容量限制
def con19_rule(model,a,k):
    if k==0:
        return pyo.Constraint.Skip
    else:
        sum_gu=sum(model.u_g[a,s,k]-model.u_g[a,s,k-1] for s in model.NS)
        sum_eu=sum(model.u_e[a,s,c,e,k]-model.u_e[a,s,c,e,k-1] for s in model.NS for c in model.C for e in model.E)
        return sum_gu+sum_eu <= Para.Q_in(a, k)
model.con17=pyo.Constraint(model.A_AC,model.K,rule=con19_rule)
        
#EV交通需求满足
def con20_rule(model,a,s,c,e,k):
    return model.u_e[a,s,c,e,k] == Para.Demand_e(a, s, c, e, k)
model.con20=pyo.Constraint(model.AR,model.NS,model.C,model.E,model.K,rule=con20_rule)

#EV守恒定律
def con21_rule(model,i,s,c,e,k):
    if e not in model.E:
        return pyo.Constraint.Skip
    else:
        return sum(model.v_e[a, s, c, e, k] for a in Para.EV_dict_in[i]) == sum(model.u_e[b, s, c, e, k] for b in Para.EV_dict_out[i])
model.con21=pyo.Constraint(model.N-model.NR,model.NS,model.C,model.E,model.K,rule=con21_rule)

def cons_sink_link_node_e_rule(model, a, s, c, e, k):
    if e not in model.E:
        return pyo.Constraint.Skip
    sink_links = Para.EV_dict_in[s]
    if sink_links[0] != a:#如果这个sink link 与这个sink不相连，那么这个u[a,s,k]==0（借来用用~~~~）
        return model.u_e[a, s, c, e, k] == 0
    else:
        return pyo.Constraint.Skip
model.cons_sink_link_node_e = pyo.Constraint(model.AS, model.NS, model.C, model.E, model.K, rule=cons_sink_link_node_e_rule)


#充电链路守恒定律
def con22_rule(model,a,s,c,e,k):
    if (k == 1) or (k == 0) or (e not in model.E):
        return model.dx[a,s,c,e,k] == 0
    else:
        term = (model.u_e[a,s,c,e,k-1]-model.u_e[a,s,c,e,k-2])-(model.v_e[a,s,c,e,k-1]-model.v_e[a,s,c,e,k-2])
        return model.dx[a,s,c,e,k] == model.x[a,s,c,e,k-1] + term
model.con22=pyo.Constraint(model.AC,model.NS,model.C,model.E,model.K,rule=con22_rule)

#充电过程建模
def con23_rule(model,a,s,c,e,k):
    if e not in model.E:
        return model.x[a, s, c, e, k] == 0
    if e == Para.max_EL:
        return model.x[a, s, c, e, k] == sum(model.dx[a, s, c, e - l, k] for l in range(Para.Alpha(a, k)+1))
    elif e > Para.Alpha(a, k):
        return model.x[a, s, c, e, k] == model.dx[a, s, c, e - Para.Alpha(a, k), k]
    else: 
        return model.x[a, s, c, e, k] == 0
model.con23=pyo.Constraint(model.AC,model.NS,model.C,model.E,model.K,rule=con23_rule)

#流限制
def con24_rule(model,a,s,c,e,k):
    if (k == 0) or (e not in model.E):
        return pyo.Constraint.Skip
    else:
        return model.v_e[a,s,c,e,k] - model.v_e[a,s,c,e,k-1] <= model.x[a,s,c,e,k]
model.con24=pyo.Constraint(model.AC,model.NS,model.C,model.E,model.K,rule=con24_rule)

#充电器数量约束
def con25_rule(model,a,k):
    return sum(model.u_e[a,s,c,e,k]-model.v_e[a,s,c,e,k] for s in model.NS for c in model.C for e in model.E)<=Para.N(a)
model.con25=pyo.Constraint(model.AC,model.K,rule=con25_rule)

#EV的u,v必不递减
def con26_rule(model,a,s,c,e,k):
    if (k == 0) or (e not in model.E):
        return model.v_e[a,s,c,e,k] == 0
    else:
        return model.v_e[a,s,c,e,k] >= model.v_e[a,s,c,e,k-1]
model.con26=pyo.Constraint(model.A,model.NS,model.C,model.E,model.K,rule=con26_rule)

def con27_rule(model,a,s,c,e,k):
    if (k == 0) or (e not in model.E):
        return model.u_e[a,s,c,e,k] == 0
    else:
        return model.u_e[a,s,c,e,k] >= model.u_e[a,s,c,e,k-1]
model.con27=pyo.Constraint(model.A,model.NS,model.C,model.E,model.K,rule=con27_rule)

#supply constraint
def cons_sink_v_g_rule(model, a, s, k):
    return model.v_g[a, s, k] == 0
model.cons_sink_v_g =pyo.Constraint(model.AS, model.NS, model.K, rule=cons_sink_v_g_rule)
def cons_sink_v_e_rule(model, a, s, c, e, k):
    return model.v_e[a, s, c, e, k] == 0
model.cons_sink_v_e = pyo.Constraint(model.AS, model.NS, model.C, model.E, model.K, rule=cons_sink_v_e_rule)

instance = model.create_instance()
opt = SolverFactory('cplex')
results = opt.solve(instance, tee=True)
instance.solutions.load_from(results)

# pyutilib.services.TempfileManager.tempdir = 'C:/user_writable_path'
#opt = SolverFactory('cplex')
#opt.options["LPMethod"] = 4
#results = opt.solve(instance, symbolic_solver_labels=True, tee=True)
#instance.solutions.load_from(results)
#instance.con20.pprint()

def pyomo_fastprocess():
    # objective_dataframe = pandas.DataFrame(data=[value(instance.OBJ)], columns=['Value'])
    # objective_dataframe.to_csv('examples/traffic/example%s/results/Objective_v%s.csv' % (TN.exam, version), sep=";")
    for v in instance.component_objects(pyo.Var, active=True):
        v = str(v)
        DictValues = eval('instance.' + v + '.extract_values()')
        vdf = pandas.DataFrame(data=DictValues, index=[0])  # 转为multiidex的dataframe
        # vdf.to_csv('Results/testNew_%s.csv' %v, sep=";", index=False)  # Write row names (index)
        vdf.to_csv('Results%s/Var_%s.csv' % (Para.exam, v), sep=";", index=False)

#pyomo_fastprocess()

# instance.pprint()
# #instance.pprint()5
#def pyomo_postprocess(options=None, instance=None, results=None):
def pyomo_postprocess():
    objective_dataframe = pandas.DataFrame(data=[pyo.value(instance.obj)], columns=['Value'])
    objective_dataframe.to_csv('Results%s/Objective.csv' %(Para.exam), sep=";")
    # objective_dataframe.to_csv('examples/cyclic_Net/Objective_%s.csv' % version, sep=";")
    rows = []
    variable_names = []
    for v in instance.component_objects(pyo.Var, active=True):
        v = str(v)
        variable_names.append(v)
        varobject = getattr(instance, str(v))
        for index in varobject:
            df = [v, index[-1], index[0:-1], varobject[index].value]
            rows.append(df)
            #if type(index) is tuple and type(index[0]) is not int:
                    #      df = [v, index, varobject[index].value]
            #    df = [v, index[0], index[1:], varobject[index].value]
            #    rows.append(df)
            #else:
            #    df = [v, 0, index, varobject[index].value]
                #print(df)
            #    rows.append(df)
    # print(rows[1][1][3])
    # print(variable_names)
    # df = pd.DataFrame(rows, columns=['Variable', 'Index', 'Value'])
    df = pandas.DataFrame(rows, columns=['Variable', 'Unit', 'Index', 'Value'])
    # df2 = df.sort(columns=['Variable', 'Index'])
    # print(df['Variable'])
    # print(df2)
    gbl = globals()
    for vname in variable_names:
        subsetting = []
        for index, rows in df.iterrows():
            #        print(rows['Variable'])
            if vname == rows['Variable']:#分离出不同的变量（x和y）
                subsetting.append(index)
                # print(subsetting)
                #    print(df.loc[subsetting])
        gbl['df_' + vname] = df.loc[subsetting]
        gbl['df_' + vname] = pandas.pivot_table(gbl['df_' + vname], values='Value', index=['Variable', 'Index'],
                                            columns='Unit')
        gbl['df_' + vname].to_csv('Results%s/Data_%s.csv' % (Para.exam, vname), sep=";")
        # gbl['df_' + vname].to_csv('examples/cyclic_Net/Data_%s_%s.csv' % (vname, version), sep=";")

pyomo_postprocess()






