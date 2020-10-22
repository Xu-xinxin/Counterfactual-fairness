"""This module is used to load different datasets """

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

#unfair unaware model

#unfair unaware model

#fair, non-deterministic model

#fair,deterministic model


#数据预处理
def load_csv(path):
    # 导入数据集
    data_read = pd.read_csv(path,usecols=[1,2,3,4,6])
    data = data_read.iloc[:,:-1].values
    y = data_read.iloc[:,4]

    #对分类数据进行处理
    data_le = LabelEncoder()
    data[:, 0] = data_le.fit_transform(data[:, 0])  # 告诉程序要处理哪一列数据(编码处理)
    ohe = OneHotEncoder(categorical_features=[0,1]) # 通过处理，使分类变量，变成有意义的数据（独热处理）
    data = ohe.fit_transform(data).toarray()

    # 将数据集拆分为训练集和测试集
    data_train, data_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=0)
    return data, data_train, data_test, y_train, y_test


if __name__=="__main__":
    print("main")
    #获取预处理后的数据
    data, data_train, data_test, y_train, y_test = load_csv('law_data.csv')
    df = pd.DataFrame(data)
    data_train = pd.DataFrame(data_train)
    data_test = pd.DataFrame(data_test)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    #对dataFrame数据加列名
    ## Amerindi-0,Asian-1,Black-2,Hispanic-3,Mexican-4,other-5,Puertorican-6,White-7
    df.columns = ['ra_Ame','ra_Asi', 'ra_Bla','ra_His','ra_Mex','ra_Oth','ra_Pue','rac_Whi','sex_1', 'sex_2','LSAT', 'UGPA']
    data_train.columns = ['ra_Ame','ra_Asi', 'ra_Bla','ra_His','ra_Mex','ra_Oth','ra_Pue','rac_Whi','sex_1', 'sex_2','LSAT', 'UGPA']
    data_test.columns = ['ra_Ame','ra_Asi', 'ra_Bla','ra_His','ra_Mex','ra_Oth','ra_Pue','rac_Whi','sex_1', 'sex_2','LSAT', 'UGPA']

    print(df)
    print(data_test)

    #统计训练集和测试集数
    print(len(df))
    print(len(data_test))
    print(len(data_train))
    #print(df.iloc[0:351,0:8])



















