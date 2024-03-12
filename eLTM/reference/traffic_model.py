#%%等同于centrailzed_decision里面的traffic_model
##增加了u_g在ac上的约束
from pyomo.environ import *
import pandas
from pyomo.opt import SolverFactory
import pickle


def build_model(TN, num_time_step, dict_E, p_ev, cum_demand_e, cum_demand_g, lamda, version):
    model = AbstractModel()
    model.K = RangeSet(0, num_time_step - 1)  # start from 0 [0 : 10]
    # 每车辆类型对应的最大能量等级
    # dict_E[1] = list(range(1, max_num_levels + 1))  # 电动车1
    # dict_E[2] = list(range(1, 6 + 1))  # 电动车1
    max_num_levels = max([max(a) for a in dict_E.values()])
    # num_ev_classes = len(dict_E.keys())

    ################################################
    # ----------------SETS--------------------------
    ################################################
    model.E = RangeSet(1, max_num_levels)
    model.C = Set(initialize=dict_E.keys())
    model.AR = Set(initialize=TN.set_l_source)
    model.AS = Set(initialize=TN.set_l_sink)
    model.AG = Set(initialize=TN.set_l_general)
    model.AC = Set(initialize=TN.set_l_charging)
    model.A = Set(initialize=TN.set_links)
    model.N = Set(initialize=TN.set_nodes)  #左闭右闭
    #model.T = RangeSet(1, num_time_step-1)#start from 1 [1 : 10]
    # model.T = RangeSet(1, num_time_step-1)#start from 1 [1 : 10]
    # model.K = RangeSet(0, num_time_step - 1)  # start from 0 [0 : 10]
    model.S = Set(initialize=TN.set_n_sink)
    model.R = Set(initialize=TN.set_n_source)
    # model.CN = Set(initialize=TN.set_n_charging)
    #################################################################
    # -------------------------PARAMETERS-----------------------------
    ##################################################################
    # model.tau = Param(initialize=TN.tau)#0.1h = 6min
    model.chi = Param(initialize=1)#10$/h = 1$/6min
    model.p_ev = Param(initialize=p_ev)#每辆电动车的充电功率：0.05MWW

    def lamda_ini(model, a, k):
        if lamda is None:
            return 0 #默认充电费用为0
        else:
            return lamda.loc[a, k] #统一单位，传入的lamda单位是/kW6min
    model.lamda = Param(model.AC, model.K, initialize=lamda_ini)
    #model.demand = Param(model.AR, model.S, model.K, initialize=cum_demand)


####################################################################
#-------------------------Vars-------------------------------------
####################################################################
    model.A_AC = model.A - model.AC
    model.v_g = Var(model.A_AC, model.S, model.K, domain=NonNegativeReals)
    model.v_e = Var(model.A, model.S, model.C, model.E, model.K, domain=NonNegativeReals)
    model.u_g = Var(model.A_AC, model.S, model.K, domain=NonNegativeReals)
    model.u_e = Var(model.A, model.S, model.C, model.E, model.K, domain=NonNegativeReals)
    model.x = Var(model.AC, model.S, model.C, model.E, model.K, domain=NonNegativeReals)
    model.dx = Var(model.AC, model.S, model.C, model.E, model.K, domain=NonNegativeReals)


#################################################################
#-----------------------objective function----------------------
##################################################################
    model.A_AS = model.A - model.AS
    model.A_AC_AS = model.A_AC - model.AS
    def obj_expression(model):
        #gasoline
        sum_g = sum(model.u_g[a, s, k] - model.v_g[a, s, k] for a in model.A_AC_AS for s in model.S for k in model.K) * model.chi
        #EVs
        sum_e = sum(model.u_e[a, s, c, e, k] - model.v_e[a, s, c, e, k] for a in model.A_AS for s in model.S for c in \
                    model.C for e in dict_E[c] for k in model.K) * model.chi
        #charging fee
        sum_cs = sum((model.u_e[a, s, c, e, k] - model.v_e[a, s, c, e, k]) * model.lamda[a, k] for a in model.AC for s \
                     in model.S for c in model.C for e in dict_E[c] for k in model.K) * model.p_ev
        return sum_g + sum_e + sum_cs
    model.OBJ = Objective(rule=obj_expression, sense=minimize)

#################################################################
#-------------------------constraints----------------------------
#################################################################



    def cons_send_max_s_g_rule(model, a, s, k):
        if k == 0:
            return Constraint.Skip
        t = k - TN.getV(a)
        if t < 0:
            return Constraint.Skip
        else:
            return model.v_g[a, s, k] <= model.u_g[a, s, t]
    model.cons_send_max_s_g = Constraint(model.A_AC, model.S, model.K, rule=cons_send_max_s_g_rule)

    def cons_u_g_rule(model, a, s, k):
        return model.u_g[a, s, k] == 0
    # model.cons_u_g = Constraint(model.AC, model.S, model.K, rule=cons_u_g_rule)


    def cons_send_max_s_e_rule(model, a, s, c, e, k):
        if (k == 0) | (e not in dict_E[c]):
            return Constraint.Skip
        t = k - TN.getV(a)
        if t < 0:
            return Constraint.Skip
        original_e = e + TN.getL(a)
        max_el_c = dict_E[c][-1]
        if e > max_el_c - TN.getL(a):
            return model.v_e[a, s, c, e, k] == 0
        else:
            return model.v_e[a, s, c, e, k] <= model.u_e[a, s, c, original_e, t]
    model.cons_send_max_s_e= Constraint(model.A_AC, model.S, model.C, model.E, model.K, rule=cons_send_max_s_e_rule)


    def cons_send_max_c_rule(model, a, k):
        if k == 0:
            return Constraint.Skip
        else:
            sum_g = sum(model.v_g[a, s, k] - model.v_g[a, s, k-1] for s in model.S)
            sum_e = sum(model.v_e[a, s, c, e, k] - model.v_e[a, s, c, e, k-1] for s in model.S for c in model.C for e in dict_E[c])
            sum_v = sum_g + sum_e
            # return sum_v <= param_max_Q_out_c.loc
            return sum_v <= TN.getMax_Q_out_c(a, 0)
    model.cons_send_max_c = Constraint(model.A_AC, model.K, rule=cons_send_max_c_rule)

    def cons_receive_remaining_spaces_rule(model, a, k):
        t = k - TN.getW(a)
        if t < 0:
            return Constraint.Skip
        else:
            sum_g = sum(model.u_g[a, s, k] - model.v_g[a, s, t] for s in model.S)
            sum_e = sum(model.u_e[a, s, c, e, k] - model.v_e[a, s, c, e, t] for s in model.S for c in model.C for e in dict_E[c])
            sum_v = sum_g + sum_e
            return sum_v <= TN.getMax_N(a)
    model.cons_receive_remaining_spaces = Constraint(model.A_AC, model.K, rule=cons_receive_remaining_spaces_rule)

    def cons_receive_max_q_rule(model, a, k):
        if k == 0:
            return Constraint.Skip
        else:
            sum_g = sum(model.u_g[a, s, k] - model.u_g[a, s, k-1] for s in model.S)
            sum_e = sum(model.u_e[a, s, c, e, k] - model.u_e[a, s, c, e, k-1] for s in model.S for c in model.C for e in dict_E[c])
            sum_v = sum_g + sum_e
            # return  sum_v <= param_max_Q_in_q.loc[a, str(k)]
            return sum_v <= TN.getMax_Q_in_q(a, 0)
    model.cons_receive_max_q = Constraint(model.A_AC, model.K, rule=cons_receive_max_q_rule)

    ##demand satisfication
    def cons_demand_g_rule(model, a, s, k):
        if (a, s) not in cum_demand_g.index:
            return model.u_g[a, s, k] == 0
        return model.u_g[a, s, k] == cum_demand_g.loc[(a, s), str(k)]

    model.cons_demand_g = Constraint(model.AR, model.S, model.K, rule=cons_demand_g_rule)

    def cons_demand_e_rule(model, a, s, c, e, k):
        if (e not in dict_E[c]) or ((a, s, c, e) not in cum_demand_e.index) or (k == 0):
            return model.u_e[a, s, c, e, k] == 0
        else:
            return model.u_e[a, s, c, e, k] == cum_demand_e.loc[(a, s, c, e), str(k)]

    model.cons_demand_e = Constraint(model.AR, model.S, model.C, model.E, model.K, rule=cons_demand_e_rule)


    ##conservation law
    # model.N_S_R = model.N - model.S - model.R
    model.N_R = model.N- model.R

    def cons_conservation_g_rule(model, i, s, k):
        return sum(model.v_g[a, s, k] for a in TN.GV_dict_in[i]) == sum(model.u_g[b, s, k] for b in TN.GV_dict_out[i])
    model.cons_conservation_g = Constraint(model.N_R, model.S, model.K, rule=cons_conservation_g_rule)

    def cons_conservation_e_rule(model, i, s, c, e, k):
        if e not in dict_E[c]:
            return Constraint.Skip
        else:
            return sum(model.v_e[a, s, c, e, k] for a in TN.EV_dict_in[i]) == sum(model.u_e[b, s, c, e, k] for b in TN.EV_dict_out[i])
    model.cons_conservation_e = Constraint(model.N_R, model.S, model.C, model.E, model.K, rule=cons_conservation_e_rule)



    ##sink_link connectes with sink_node
    def cons_sink_link_node_g_rule(model, a, s, k):
        sink_links = TN.GV_dict_in[s]
        if sink_links[0] != a:#如果这个sink link 与这个sink不相连，那么这个u[a,s,k]==0
            return model.u_g[a, s, k] == 0
        else:
            return Constraint.Skip
    model.cons_sink_link_node_g = Constraint(model.AS, model.S, model.K, rule=cons_sink_link_node_g_rule)

    def cons_sink_link_node_e_rule(model, a, s, c, e, k):
        if e not in dict_E[c]:
            Constraint.Skip
        sink_links = TN.EV_dict_in[s]
        if sink_links[0] != a:#如果这个sink link 与这个sink不相连，那么这个u[a,s,k]==0
            return model.u_e[a, s, c, e, k] == 0
        else:
            return Constraint.Skip
    model.cons_sink_link_node_e = Constraint(model.AS, model.S, model.C, model.E, model.K, rule=cons_sink_link_node_e_rule)

    ##charging links
    def cons_charging_link_occupancy_rule(model, a, s, c, e, k):
        if (k == 0) or (k == 1) or (e not in dict_E[c]):
            return model.dx[a, s, c, e, k] == 0
        else:
            return model.dx[a, s, c, e, k] == model.x[a, s, c, e, k-1] + model.u_e[a, s, c, e, k-1] - model.u_e[a, s, c, e, k-2]-\
                                  model.v_e[a, s, c, e, k - 1] + model.v_e[a, s, c, e, k - 2]
    model.cons_charging_link_occupancy = Constraint(model.AC, model.S, model.C, model.E, model.K, rule=cons_charging_link_occupancy_rule)

    ## alpha is natural number []0,1,2,3...]
    def cons_update_charging_alpha_rule(model, a, s, c, e, k):
        if e not in dict_E[c]:
            return model.x[a, s, c, e, k] == 0
        if e == dict_E[c][-1]:##maximum EL
            return model.x[a, s, c, e, k] == sum(model.dx[a, s, c, e - l, k] for l in range(TN.getAlpha(a, 0)+1))
        #elif l > param_alpha.loc[i, str(t)]:
        elif e > TN.getAlpha(a, 0): #alpha <= e < maximum
            return model.x[a, s, c, e, k] == model.dx[a, s, c, e - TN.getAlpha(a, 0), k]
        else: #！！l <= param_alpha.loc[i, str(t)]
            return model.x[a, s, c, e, k] == 0
    model.cons_update_charging_alpha = Constraint(model.AC, model.S, model.C, model.E, model.K, rule=cons_update_charging_alpha_rule)


    def cons_charging_link_send_rule(model, a, s, c, e, k):
        if (k == 0) or (e not in dict_E[c]):
            return Constraint.Skip
        else:
            return model.v_e[a, s, c, e, k] - model.v_e[a, s, c, e, k-1] <= model.x[a, s, c, e, k]
    model.cons_charging_link_send = Constraint(model.AC, model.S, model.C, model.E, model.K, rule=cons_charging_link_send_rule)

    def cons_charging_link_recieve_rule(model, a, k):
        return sum(model.u_e[a, s, c, e, k] - model.v_e[a, s, c, e, k] for s in model.S for c in model.C for e in dict_E[c])\
               <= TN.getMax_N(a)
    model.cons_charging_link_recieve = Constraint(model.AC, model.K, rule=cons_charging_link_recieve_rule)


    ##nonnegative and ini when k=0
    def cons_nonnegative_u_g_rule(model, a, s, k):
        if k == 0:
            return model.u_g[a, s, k] == 0
        else:
            return model.u_g[a, s, k] - model.u_g[a, s, k-1] >= 0
    model.cons_nonnegative_u_g = Constraint(model.A_AC, model.S, model.K, rule=cons_nonnegative_u_g_rule)

    def cons_nonnegative_v_g_rule(model, a, s, k):
        if k == 0:
            return model.v_g[a, s, k] == 0
        else:
            return model.v_g[a, s, k] - model.v_g[a, s, k - 1] >= 0
    model.cons_nonnegative_v_g = Constraint(model.A_AC, model.S, model.K, rule=cons_nonnegative_v_g_rule)

    def cons_nonnegative_u_e_rule(model, a, s, c, e, k):
        if (e not in dict_E[c]) or (k == 0):
            return model.u_e[a, s, c, e, k] == 0
        else:
            return model.u_e[a, s, c, e, k] - model.u_e[a, s, c, e, k-1] >= 0
    model.cons_nonnegative_u_e = Constraint(model.A, model.S, model.C, model.E, model.K, rule=cons_nonnegative_u_e_rule)

    def cons_nonnegative_v_e_rule(model, a, s, c, e, k):
        if (e not in dict_E[c]) or (k == 0):
            return model.v_e[a, s, c, e, k] == 0
        else:
            return model.v_e[a, s, c, e, k] - model.v_e[a, s, c, e, k - 1] >= 0
    model.cons_nonnegative_v_e = Constraint(model.A, model.S, model.C, model.E, model.K, rule=cons_nonnegative_v_e_rule)

    ##sink link的流出为0
    def cons_sink_v_g_rule(model, a, s, k):
        return model.v_g[a, s, k] == 0
    model.cons_sink_v_g = Constraint(model.AS, model.S, model.K, rule=cons_sink_v_g_rule)
    def cons_sink_v_e_rule(model, a, s, c, e, k):
        return model.v_e[a, s, c, e, k] == 0
    model.cons_sink_v_e = Constraint(model.AS, model.S, model.C, model.E, model.K, rule=cons_sink_v_e_rule)

    ##能量等级大于等于（最大能量等级-走过这段路需要消耗的能量等级+1）的流出为0
    def cons_sink_v_e_0_rule(model, a, s, c, e, k):
        max_el_c = dict_E[c][-1]
        if e >= max_el_c - TN.getL(a)+1:
            return model.v_e[a, s, c, e, k] == 0
        else:
            return Constraint.Skip
    # model.cons_sink_v_e_0 = Constraint(model.A, model.S, model.C, model.E, model.K, rule=cons_sink_v_e_0_rule)


    # def cons_ini_u_rule(model, i, j, s):
    #     model.u[i, j, s, 0] == 0
    # model.cons_ini_u = Constraint(model.A, model.S, rule=cons_ini_u_rule)
    #
    # def cons_ini_v_rule(model, i, j, s):
    #     model.v[i, j, s, 0] == 0
    # model.cons_ini_v = Constraint(model.A, model.S, rule=cons_ini_v_rule )

    ##for EVs



    instance = model.create_instance()
    # opt = SolverFactory('glpk')
    # results = opt.solve(instance, tee=True)
    # instance.solutions.load_from(results)

    # pyutilib.services.TempfileManager.tempdir = 'C:/user_writable_path'
    opt = SolverFactory('gurobi')
    opt.options["LPMethod"] = 4
    results = opt.solve(instance, symbolic_solver_labels=True, tee=True)
    instance.solutions.load_from(results)

    # def pyomo_fastprocess():
    #     # objective_dataframe = pandas.DataFrame(data=[value(instance.OBJ)], columns=['Value'])
    #     # objective_dataframe.to_csv('examples/traffic/example%s/results/Objective_v%s.csv' % (TN.exam, version), sep=";")
    #     for v in instance.component_objects(Var, active=True):
    #         v = str(v)
    #         DictValues = eval('instance.' + v + '.extract_values()')
    #         vdf = pandas.DataFrame(data=DictValues, index=[0])  # 转为multiidex的dataframe
    #         # vdf.to_csv('Results/testNew_%s.csv' %v, sep=";", index=False)  # Write row names (index)
    #         vdf.to_csv('examples/traffic/example%s/results/Var_%s_%s.csv' % (TN.exam, v, version), sep=";", index=False)

    # pyomo_fastprocess()

    # # instance.pprint()
    # # #instance.pprint()5
    # #def pyomo_postprocess(options=None, instance=None, results=None):
    # def pyomo_postprocess():
    #     objective_dataframe = pandas.DataFrame(data=[value(instance.OBJ)], columns=['Value'])
    #     objective_dataframe.to_csv('examples/traffic/example%s/results/Objective_v%s_old.csv' %(TN.exam, version), sep=";")
    #     # objective_dataframe.to_csv('examples/cyclic_Net/Objective_%s.csv' % version, sep=";")
    #     rows = []
    #     variable_names = []
    #     for v in instance.component_objects(Var, active=True):
    #         v = str(v)
    #         variable_names.append(v)
    #         varobject = getattr(instance, str(v))
    #         for index in varobject:
    #             df = [v, index[-1], index[0:-1], varobject[index].value]
    #             rows.append(df)
    #             #if type(index) is tuple and type(index[0]) is not int:
    #                     #      df = [v, index, varobject[index].value]
    #             #    df = [v, index[0], index[1:], varobject[index].value]
    #             #    rows.append(df)
    #             #else:
    #             #    df = [v, 0, index, varobject[index].value]
    #                 #print(df)
    #             #    rows.append(df)
    #     # print(rows[1][1][3])
    #     # print(variable_names)
    #     # df = pd.DataFrame(rows, columns=['Variable', 'Index', 'Value'])
    #     df = pandas.DataFrame(rows, columns=['Variable', 'Unit', 'Index', 'Value'])
    #     # df2 = df.sort(columns=['Variable', 'Index'])
    #     # print(df['Variable'])
    #     # print(df2)
    #     gbl = globals()
    #     for vname in variable_names:
    #         subsetting = []
    #         for index, rows in df.iterrows():
    #             #        print(rows['Variable'])
    #             if vname == rows['Variable']:#分离出不同的变量（x和y）
    #                 subsetting.append(index)
    #                 # print(subsetting)
    #                 #    print(df.loc[subsetting])
    #         gbl['df_' + vname] = df.loc[subsetting]
    #         gbl['df_' + vname] = pandas.pivot_table(gbl['df_' + vname], values='Value', index=['Variable', 'Index'],
    #                                             columns='Unit')
    #         gbl['df_' + vname].to_csv('examples/example%s/results/Data_%s_%s.csv' % (TN.exam, vname, version), sep=";")
    #         # gbl['df_' + vname].to_csv('examples/cyclic_Net/Data_%s_%s.csv' % (vname, version), sep=";")

    # # pyomo_postprocess()





    # def save_aggregation_e(v):
    #     version = v
    #     objective_dataframe = pandas.DataFrame(data=[value(instance.OBJ)], columns=['Value'])
    #     objective_dataframe.to_csv('examples/traffic/example%s/results/Objective_%s.csv' %(TN.exam, version), sep=";")
    #     x_dim = ['id', 'destinations', 'class', 'levels', 'time']
    #     var_x_dim_lst = list(range(len(x_dim)))
    #     DictValues = instance.u_e.extract_values()
    #     stage1_x = pandas.DataFrame(data=DictValues, index=[0])  # 转为multiidex的dataframe
    #     # 把索引由字符转为int数据类型
    #     stage1_x.columns.set_levels([stage1_x.columns.levels[i].astype(int) for i in var_x_dim_lst], level=var_x_dim_lst,
    #                             inplace=True)
    #     stage1_x.rename_axis(x_dim, axis='columns', inplace=True)  # 设置每个level的名字
    #     stage1_x = stage1_x.T  # 1.旋转表
    #     stage1_x.reset_index(inplace=True)  # 2重置索引
    #     stage1_x.set_index(x_dim[0:-1], inplace=True)  # 设id为索引
    #     stage1_x_tab = stage1_x.pivot(columns=x_dim[-1])  # 设时间为列名
    #     stage1_x_tab = stage1_x_tab.droplevel(0, axis=1)  # 去掉莫名多出来的一层索引0
    #     stage1_agu = stage1_x_tab.sum(level='id')
    #     stage1_agu.to_csv('examples/traffic/example%s/results/Data_agu_%s.csv' % (TN.exam, version), sep=";", index=True)
    #     stage1_axR = stage1_x_tab.sum(level=['id', 'destinations'])
    #     # stage1_axR.to_csv('examples/example%s/results/Data_v%s_auS.csv' % (TN.exam, version), sep=";", index=True)
    #     stage1_axL = stage1_x_tab.sum(level=['id', 'levels'])
    #     # stage1_axL.to_csv('examples/traffic/example%s/results/Data_auL_%s.csv' % (TN.exam, version), sep=";", index=True)

    #     DictValues_v = instance.v_e.extract_values()
    #     stage1_x = pandas.DataFrame(data=DictValues_v, index=[0])  # 转为multiidex的dataframe
    #     # 把索引由字符转为int数据类型
    #     stage1_x.columns.set_levels([stage1_x.columns.levels[i].astype(int) for i in var_x_dim_lst], level=var_x_dim_lst,
    #                                 inplace=True)
    #     stage1_x.rename_axis(x_dim, axis='columns', inplace=True)  # 设置每个level的名字
    #     stage1_x = stage1_x.T  # 1.旋转表
    #     stage1_x.reset_index(inplace=True)  # 2重置索引
    #     stage1_x.set_index(x_dim[0:-1], inplace=True)  # 设id为索引
    #     stage1_x_tab = stage1_x.pivot(columns=x_dim[-1])  # 设时间为列名
    #     stage1_x_tab = stage1_x_tab.droplevel(0, axis=1)  # 去掉莫名多出来的一层索引0
    #     stage1_agv = stage1_x_tab.sum(level='id')
    #     # stage1_agv.to_csv('examples/traffic/example%s/results/Data_agv_%s.csv' % (TN.exam, version), sep=";", index=True)
    #     # stage1_axR = stage1_x_tab.sum(level=['id', 'destinations'])
    #     # stage1_axR.to_csv('examples/example%s/results/Data_v%s_avS.csv' % (exam, version), sep=";", index=True)
    #     # stage1_axL = stage1_x_tab.sum(level=['id', 'levels'])
    #     # stage1_axL.to_csv('examples/traffic/example%s/results/Data_avL_%s.csv' % (TN.exam, version), sep=";", index=True)

    #     occupancy = stage1_agu - stage1_agv
    #     occupancy.to_csv('examples/traffic/example%s/results/Data_occupenacy_ev_%s.csv' % (TN.exam, version), sep=";", index=True)
    #     return occupancy
    # occupancy_e = save_aggregation_e(version)

    # def save_aggregation_g(v):
    #     version = v
    #     x_dim = ['id', 'destinations', 'time']
    #     var_x_dim_lst = list(range(len(x_dim)))
    #     DictValues = instance.u_g.extract_values()
    #     stage1_x = pandas.DataFrame(data=DictValues, index=[0])  # 转为multiidex的dataframe
    #     # 把索引由字符转为int数据类型
    #     stage1_x.columns.set_levels([stage1_x.columns.levels[i].astype(int) for i in var_x_dim_lst], level=var_x_dim_lst,
    #                             inplace=True)
    #     stage1_x.rename_axis(x_dim, axis='columns', inplace=True)  # 设置每个level的名字
    #     stage1_x = stage1_x.T  # 1.旋转表
    #     stage1_x.reset_index(inplace=True)  # 2重置索引
    #     stage1_x.set_index(x_dim[0:-1], inplace=True)  # 设id为索引
    #     stage1_x_tab = stage1_x.pivot(columns=x_dim[-1])  # 设时间为列名
    #     stage1_x_tab = stage1_x_tab.droplevel(0, axis=1)  # 去掉莫名多出来的一层索引0
    #     stage1_agu = stage1_x_tab.sum(level='id')
    #     stage1_agu.to_csv('examples/traffic/example%s/results/Data_ug_%s.csv' % (TN.exam, version), sep=";", index=True)


    #     DictValues_v = instance.v_g.extract_values()
    #     stage1_x = pandas.DataFrame(data=DictValues_v, index=[0])  # 转为multiidex的dataframe
    #     # 把索引由字符转为int数据类型
    #     stage1_x.columns.set_levels([stage1_x.columns.levels[i].astype(int) for i in var_x_dim_lst], level=var_x_dim_lst,
    #                                 inplace=True)
    #     stage1_x.rename_axis(x_dim, axis='columns', inplace=True)  # 设置每个level的名字
    #     stage1_x = stage1_x.T  # 1.旋转表
    #     stage1_x.reset_index(inplace=True)  # 2重置索引
    #     stage1_x.set_index(x_dim[0:-1], inplace=True)  # 设id为索引
    #     stage1_x_tab = stage1_x.pivot(columns=x_dim[-1])  # 设时间为列名
    #     stage1_x_tab = stage1_x_tab.droplevel(0, axis=1)  # 去掉莫名多出来的一层索引0
    #     stage1_agv = stage1_x_tab.sum(level='id')
    #     # stage1_agv.to_csv('examples/traffic/example%s/results/Data_vg_%s.csv' % (TN.exam, version), sep=";", index=True)

    #     occupancy = stage1_agu - stage1_agv
    #     occupancy.to_csv('examples/traffic/example%s/results/Data_occupenacy_vg_%s.csv' % (TN.exam, version), sep=";", index=True)
    #     return occupancy
    # occupancy_g = save_aggregation_g(version)
    # occupancy_sum = occupancy_e + occupancy_g#因为gv没有charging links的项，相加gv+ev后，charging links上的occupancy就没了
    # occupancy_sum.to_csv('examples/traffic/example%s/results/Data_occupenacy_v_sum_%s.csv' % (TN.exam, version), sep=";",
    #                  index=True)
    # # travelCost = occupancy_sum.loc[instance.A_AS.data(), :]#除去sink links
    # # travelCost = travelCost.sum().sum()*instance.chi.value

    # def print_chargingCost():
    #     u_e = instance.u_e.extract_values()
    #     v_e = instance.v_e.extract_values()
    #     lamda = instance.lamda.extract_values()
    #     chargingCost = sum((u_e[a, s, c, e, k] - v_e[a, s, c, e, k]) * lamda[a, k] for a in TN.set_l_charging for s \
    #                  in TN.set_n_sink for c in dict_E.keys() for e in dict_E[c] for k in range(0, num_time_step)) * instance.p_ev.value
    #     objective = value(instance.OBJ)
    #     dict = {'chargingCost': chargingCost, 'travelCost': objective - chargingCost,
    #             'OBJTrafficCost': objective}
    #     stage1_x = pandas.DataFrame(data=dict, index=[0])
    #     stage1_x.to_csv('examples/traffic/example%s/results/Data_TrafficCost_%s.csv' % (TN.exam, version), sep=";", index=True)
    # print_chargingCost()
