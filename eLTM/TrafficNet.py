import pandas as pd

class TN:
    def __init__(self,tau,version,timestep) -> None:
        self.tau=tau
        self.version=version
        self.timestep=timestep
        self.period=int(tau/timestep)

    def readCSV(self):
        #一些参数
        self.maxEL=10
        self.e_class=[1,]
        
        #充电速率表格
        self.alphadf=pd.read_csv("data//alpha.csv")
        self.alphadf.set_index('id',inplace=True)

        #最大流入流出
        self.maxQindf=pd.read_csv("data//maxQin.csv")
        self.maxQindf.set_index('id',inplace=True)

        self.maxQoutdf=pd.read_csv("data//maxQout.csv")
        self.maxQoutdf.set_index('id',inplace=True)

        #需求数据
        self.demandg_df=pd.read_csv("data//demand_g.csv")
        self.demandg_df.set_index(['id_link','sink_node'],inplace=True)#这里油车没有需求，就不处理数据了

        self.demande_df=pd.read_csv("data//demand_e.csv")
        self.demande_df.set_index(['id_link','sink_node','class','el'],inplace=True)
        
        for (l,a,c,e) in self.demande_df.index:
            mylist=self.demande_df.columns.tolist()
            tep=mylist.pop(0)
            for t in mylist:                
                self.demande_df.loc[(l,a,c,e),t]+=self.demande_df.loc[(l,a,c,e),tep]
                tep=t

        #link 数据
        self.linkdf=pd.read_csv("data//link.csv")
        self.linkdf.set_index('id',inplace=True)

        self.nodedf=pd.read_csv("data//node.csv")
        self.nodedf.set_index('id',inplace=True)

        #link和node分类
        RL=[]
        SL=[]
        GL=[]
        CL=[]

        RN=[]
        SN=[]
        self.AN=self.nodedf.index.tolist()

        for i in self.linkdf.index:
            if(self.linkdf.loc[i,'type']=='R'):
                RL.append(i)
            if(self.linkdf.loc[i,'type']=='S'):
                SL.append(i)
            if(self.linkdf.loc[i,'type']=='G'):
                GL.append(i)
            if(self.linkdf.loc[i,'type']=='C'):
                CL.append(i)
        
        for i in self.nodedf.index:
            if(self.nodedf.loc[i,'type']=='R'):
                RN.append(i)
            if(self.nodedf.loc[i,'type']=='S'):
                SN.append(i)

        self.RL=RL
        self.SL=SL
        self.GL=GL
        self.CL=CL
        self.RN=RN
        self.SN=SN
        self.AL=self.linkdf.index.tolist()

        #node挂载前置link及后置link
        def Node_in_out(node):
            node_in_g=[]
            node_out_g=[]
            node_in_e=[]
            node_out_e=[]

            for i in self.linkdf.index:
                if(self.linkdf.loc[i,'end']==node):
                    node_in_e.append(i)
                    if(self.linkdf.loc[i,'start']!=self.linkdf.loc[i,'end']):
                        node_in_g.append(i)
                if(self.linkdf.loc[i,'start']==node):
                    node_out_e.append(i)
                    if(self.linkdf.loc[i,'start']!=self.linkdf.loc[i,'end']):
                        node_out_g.append(i)
            
            return node_in_g,node_out_g,node_in_e,node_out_e
        
        self.g_in={}
        self.g_out={}
        self.e_in={}
        self.e_out={}

        for i in self.AN:
            gin,gout,ein,eout=Node_in_out(i)
            self.g_in[i]=gin
            self.g_out[i]=gout
            self.e_in[i]=ein
            self.e_out[i]=eout

        #生成OD对
        OD_pair={}
        od=1
        for i in self.RL:
            for j in self.SL:
                OD_pair.update({od:(i,j)})
                od+=1
        self.OD_pair=OD_pair


        #生产交通网络图结构
        self.T_graph={}
        for i in self.linkdf.index:
            T_graph=[]
            for j in self.linkdf.index:
                if((self.linkdf.loc[i,'end']==self.linkdf.loc[j,'start']) & (self.linkdf.loc[j,'type']!='C')):
                    T_graph.append(j)
            self.T_graph.update({i:T_graph})

        #由网络图生成路径
        Rw={}
        self.paths=[]
        self.Allpaths={}
        def findPath(start,end,graph,path):
            if not path:
                path.append(start)
            if(start==end):
                self.paths.append(path)
                self.Allpaths.update({len(self.Allpaths)+1:path})
                return
            if(start!=end):
                for i in graph[start]:
                    path.append(i)
                    newpath=path[:]
                    findPath(i,end,graph,newpath)
                    path.pop()

        for i in self.OD_pair.keys():
            findPath(self.OD_pair[i][0],self.OD_pair[i][1],self.T_graph,path=[])
            Rw.update({i:self.paths})
            self.paths=[]
        
        self.Rw=Rw

        #通过link对应的正向传播时间


        #通过link的反向传播时间




    def test(self):
        #建立path表格
        # pathdata={'id':self.Allpaths.keys(),'route':self.Allpaths.values()}
        # pathdf=pd.DataFrame(pathdata)
        # pathdf.to_csv('data//path.csv',index=None) 
        # print("self.alphadf")
        # print(self.alphadf)
        # print("self.maxQindf")
        # print(self.maxQindf)
        # print("self.maxQoutdf")
        # print(self.maxQoutdf)
        # print("self.demandg_df")
        # print(self.demandg_df)
        print("self.demande_df")
        print(self.demande_df)
        # print("self.linkdf")
        # print(self.linkdf)
        # print(self.RL,self.SL,self.GL,self.CL,self.AL,self.RN,self.SN,self.AN)
        # print(self.g_in,'\n',self.g_out,'\n',self.e_in,'\n',self.e_out)
        # print(self.OD_pair,"\n",self.T_graph)
        # print(self.Rw,'\n',self.Allpaths)

# one=TN(60,1,2)
# one.readCSV()
# one.test()
