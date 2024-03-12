# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 13:00:34 2022

@author: zhang ke
"""

import pandas
import math#就只用一下ceil函数

class TN:
    def __init__(self, tau, exam, timestep_num):
        
        self.tau=tau
        self.exam=exam
        self.step_num=timestep_num
        
        ######
        #读取alpha文件数据及处理
        ######
        self.file_alpha=pandas.read_csv('Data%s/alpha_e.csv' %exam ,sep=';')
        self.file_alpha.set_index('id',inplace=True)
        
        ########
        #读取demand文件数据及处理
        ########
        self.file_demandg=pandas.read_csv('Data%s/demand_g.csv' %exam, sep=';')
        self.file_demandg.set_index(['id','sink'],inplace=True)
        self.demand_g=self.file_demandg.cumsum(axis=1)
        
        self.file_demande=pandas.read_csv('Data%s/demand_e1.csv' %exam, sep=';')
        self.max_EL=10
        self.file_demande.set_index(['id_link','sink_node','class','el'],inplace=True)
        self.demand_e=self.file_demande.cumsum(axis=1)
        
        ##########
        #读取link文件数据并处理
        ##########
        self.file_link=pandas.read_csv('Data%s/links.csv' %exam, sep=';')
        n2n_links = self.file_link[['start', 'end', 'id']]
        #将参数v,w等分类输出
        
        self.file_link.set_index('id',inplace=True)
        
        #分类link和node
        
        source_link=set()
        sink_link=set()
        charging_link=set()
        general_link=set()
        source_node=set()
        sink_node=set()
        
        for link in self.file_link.index:
            if self.file_link.loc[link]['type']=='R':
                source_link.add(link)
                source_node.add(self.file_link.loc[link]['start'])
            if self.file_link.loc[link]['type']=='S':
                sink_link.add(link)
                sink_node.add(self.file_link.loc[link]['end'])
            if self.file_link.loc[link]['type']=='G':
                general_link.add(link)
            if self.file_link.loc[link]['type']=='C':
                charging_link.add(link)
        
        link = source_link | sink_link | charging_link | general_link
        start_node=set(self.file_link['start'].values)
        end_node=set(self.file_link['end'].values)
        node = start_node.union(end_node)
        
        self.linknum=len(self.file_link.index)
        self.RL=source_link
        self.GL=general_link
        self.CL=charging_link
        self.SL=sink_link
        self.AL=link
        
        self.AN=node
        self.RN=source_node
        self.SN=sink_node
        
        #这里的邻接表构建就先借鉴老师的了（狗头）
        
        n2n_links = n2n_links.groupby(['start', 'end'])
        n2n_links = n2n_links['id'].unique()
        n2n_links = n2n_links.to_frame()
        n2n_links = n2n_links['id']
        
        def Nodes_init(node):
            # input: id of node
            # output:
            # -set_out: set of outgoing links of the node
            # -set_in: set of entering links of the node
            EV_set_out = []
            EV_set_in = []
            GV_set_out = []
            GV_set_in = []
            for (i, j) in n2n_links.index:
                if i == node:
                    set_outLinks = n2n_links.loc[(i, j)].tolist()#return set of outgoing links
                    EV_set_out.extend(set_outLinks)
                    for a in set_outLinks:#删除掉charging links
                        if a in self.CL:
                            set_outLinks.remove(a)
                    GV_set_out.extend(set_outLinks)
                if j == node:
                    set_inLinks = n2n_links.loc[(i, j)].tolist()#return set of entering links
                    EV_set_in.extend(set_inLinks)
                    for a in set_inLinks:  # 删除掉charging links
                        if a in self.CL:
                            set_inLinks.remove(a)
                    GV_set_in.extend(set_inLinks)
            return EV_set_out, EV_set_in, GV_set_out, GV_set_in

        self.EV_dict_out = {}
        self.EV_dict_in = {}
        self.GV_dict_out = {}
        self.GV_dict_in = {}
        for i in self.AN:
            EV_set_out, EV_set_in, GV_set_out, GV_set_in = Nodes_init(i)
            self.EV_dict_out[i] = EV_set_out
            self.EV_dict_in[i] = EV_set_in
            self.GV_dict_out[i] = GV_set_out
            self.GV_dict_in[i] = GV_set_in
    
    
        #根据link.csv生成图的邻接表和od_pair
        self.Adjlists={}
        for id1 in self.file_link.index:
            Adjlist=[]
            for id2 in self.file_link.index:
                if self.file_link.loc[id1]['end'] == self.file_link.loc[id2]['start'] and id1 != id2:
                    Adjlist.append(id2)
            self.Adjlists.update({id1:Adjlist})
        
        def generate_od_pair(resoure_link,sink_link):
            od_pair={}
            od_id=1
            for o in resoure_link:
                for d in sink_link:
                    od_pair.update({od_id:(o,d)})
                    od_id=od_id+1
            return od_pair
        self.od_pair=generate_od_pair(self.RL, self.SL)
        
        self.W=list(self.od_pair.keys())
        
        #######
        #根据上述邻接表生成所有路径和Rw
        #######
        
        def findAllPath(graph, start, end, path=[]):
            if not path:
                path.append(start)
            if start == end:
                if len(path[:])<=10:
                    self.Path.update({len(self.Path)+1:path[:]})
                return
            for node in graph[start]:
                if node not in path :
                    path.append(node)
                    findAllPath(graph, node, end, path)
                    path.pop()
                    
        self.Rw={}
        self.Path={}    
        last_path_id=[]
        for w in self.W:
            (o,d)=self.od_pair[w]
            findAllPath(self.Adjlists, o, d)
            self.Rw.update({w:list(set(self.Path.keys())-set(last_path_id))})
            last_path_id=(list(self.Path.keys()))
        self.R=self.Path.keys()

        ########3
        #根据上述path生成充电站能级
        #########
        self.dict_Eo={}
        self.dict_Ed={}
        self.dict_TC={}
        self.Rg=[]
        self.Rl=[]
        self.Rw_g={}
        self.Rw_l={}
        
        def generate_EL(link,path,mode):
            ans=0
            
            if mode == 'o':
                for blink in path[:path.index(link)]:
                    ans=ans+self.L(blink)
            if mode == 'd':
                for alink in path[path.index(link):]:
                    ans=ans+self.L(alink)
            return ans
                
        for link in self.CL:
            for r in self.Path.keys():
                if link in self.Path[r]:
                    self.Rl.append(r)
                    Eo=generate_EL(link,self.Path[r], 'o')
                    Ed=generate_EL(link,self.Path[r], 'd')
                    self.dict_Ed.update({(link,r):Ed})
                    self.dict_Eo.update({(link,r):Eo})
                    for e in range(0,self.max_EL+1):
                        TC=(Eo+Ed-e)/self.Alpha(link, 2)
                        self.dict_TC.update({(link,r,e):math.ceil(TC)})
                        
        self.Rg=self.R - self.Rl
        for w in self.W:
            temp=[]
            for link in self.Rw[w]:
                if link in self.Rg:
                    temp.append(link)
            self.Rw_g.update({w:temp})
            self.Rw_l.update({w:set(self.Rw[w])-set(temp)})
                    
        self.T_w={}
        for w in self.W:
            for e in list(range(0,self.max_EL+1)):
                temp_max=-100
                for a in self.CL:
                    for r in self.Rw_l[w]:
                        if a in self.Path[r]:
                            temp_max=max(temp_max,self.dict_TC[(a,r,e)])
                self.T_w.update({(w,e):temp_max})
                
        ########
        #读取maxQ_in和maxQ_out文件的数据并处理
        ########
        
        self.file_mqin=pandas.read_csv('Data%s/max_Q_in.csv' %exam,sep=';')
        self.file_mqin.set_index('id',inplace=True)
        self.file_mqout=pandas.read_csv('Data%s/max_Q_out.csv' %exam,sep=';')
        self.file_mqout.set_index('id',inplace=True)
        
        
    def Alpha(self,a,k):
        return self.file_alpha.loc[a][k]
    
    def Demand_g(self,a,s,k):
        if (a,s) not in self.demand_g.index:
            return 0
        else:
            return self.demand_g.loc[a,s][k]
        
    def Demand_e(self,a,s,c,e,k):
        if (a,s,c,e) not in self.demand_e.index or (k==0):
            return 0
        else:
            return self.demand_e.loc[a,s,c,e][k]
        
    def V(self,a):
        return self.file_link.loc[a]['v']
    
    def W(self,a):
        return self.file_link.loc[a]['w']
    
    def N(self,a):
        return self.file_link.loc[a]['max_N']
    
    def L(self,a):
        return self.file_link.loc[a]['l']
    
    def Q_in(self,a,k):
        #此例中q并不时变
        k=0
        return self.file_mqin.loc[a][k]
    
    def Q_out(self,a,k):
        #同理q不时变
        k=0
        return self.file_mqout.loc[a][k]
    
    def EO(self,a,r):
        if (a,r) not in self.dict_Eo.keys():
            return -999
        else:
            return self.dict_Eo[(a,r)]
    
    def ED(self,a,r):
        if (a,r) not in self.dict_Ed.keys():
            return -999
        else:
            return self.dict_Ed[(a,r)]
        
    def TC(self,a,r,e):
        if (a,r,e) not in self.dict_TC.keys():
            return 1000
        else:
            return self.dict_TC[(a,r,e)]
    
    def Tw(self,w,e):
        if (w,e) not in self.T_w.keys():
            return 1000
        else:
            return self.Tw[(w,e)]
    
    
    
    
    