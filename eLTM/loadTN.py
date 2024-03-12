import pandas as pd
import numpy as np

class loadTN:
    def __init__(self) -> None:
        pass
    
    def mkCSV(self):
        loc='data'
        #创建link的数据        
        linkdata=[0,0,1,0,0,0,'R',99999,
              1,1,2,5,5,5,'G',26733,
              2,1,4,8,8,8,'G',42772.8,
              3,1,3,5,5,5,'G',26733,
              4,2,4,5,5,5,'G',26733,
              5,3,4,4,4,4,'G',21386.4,
              7,2,2,0,0,0,'C',15,
              10,3,3,0,0,0,'C',15,
              12,4,9,0,0,0,'S',99999]
        linkarr=np.array(linkdata).reshape(9,8)
        linkdf=pd.DataFrame(linkarr,columns=['id','start','end','v','w','l','type','max_N'],index=None)
        linkdf.to_csv('%s//link.csv'%loc,index=None)

        #创建node数据
        nodedata={'id':[0,1,2,3,4,9],
                  'type':['R','G','G','G','G','S']}
        nodedf=pd.DataFrame(nodedata)
        nodedf.to_csv('%s//node.csv'%loc,index=None)

        #创建充电速率alpha数据
        alphaarr=np.linspace(start=4,stop=4,num=120,dtype=np.int_).reshape(2,60)
        alphaid=np.array([7,10]).reshape(2,1)
        alpha=np.concatenate((alphaid,alphaarr),axis=1)
        columnTime=['id']+[i for i in range(60)]
        alphadf=pd.DataFrame(alpha,columns=columnTime,index=None)
        alphadf.to_csv('%s//alpha.csv'%loc,index=None)

        #创建流量maxQin maxQout数据 这里两者相同
        maxQindata=[0,1,2,3,4,5,6,7,8,9,10,11,12,9999,216,216,216,216,216,216,216,216,216,216,216,9999]
        maxQinarr=np.array(maxQindata).reshape(2,13).T
        maxQindf=pd.DataFrame(maxQinarr,columns=['id',0])
        maxQindf.to_csv('%s//maxQin.csv'%loc,index=None)
        maxQindf.to_csv('%s//maxQout.csv'%loc,index=None)

        # 创建电车需求demand_e数据
        demand_edata=[0,9,1,6,10,0,10,0,10,0,10,0,10,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,9,1,7,10,0,10,0,10,0,10,0,10,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,9,1,8,10,0,10,0,10,0,10,0,10,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,9,1,9,35,0,35,0,35,0,35,0,35,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,9,1,10,35,0,35,0,35,0,35,0,35,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    ]
        demand_earr=np.array(demand_edata).reshape(5,64)
        columndemand=['id_link','sink_node','class','el',1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
                      31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60]
        demand_edf=pd.DataFrame(demand_earr,columns=columndemand)
        demand_edf.to_csv('%s//demand_e.csv'%loc,index=None)
        
        #创建油车demand_g数据
        demand_garr=np.zeros((1,62),dtype=np.int_)
        columng=['id_link','sink_node',1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
                      31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60]
        demand_gdf=pd.DataFrame(demand_garr,columns=columng)
        demand_gdf.to_csv('%s//demand_g.csv'%loc,index=None)




# one=loadTN()
# one.mkCSV()
