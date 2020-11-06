import numpy as np
from numpy import *
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from law_school_train import nondeterministic_tr
from law_school_only_u import nondeterministic_te

#数据预处理
def load_csv(path):
    # 导入数据集
    data_read = pd.read_csv(path,usecols=[1,2,3,4,6])
    data_read = data_read.dropna()
    print(len(data_read))

    #划分自变量和因变量
    data = data_read.iloc[:,:-1].values
    y = data_read.iloc[:,4]

    #对分类数据进行处理
    data_le = LabelEncoder()
    data[:, 0] = data_le.fit_transform(data[:, 0])  # 告诉程序要处理哪一列数据(编码处理)
    # 通过处理，使分类变量，变成有意义的数据（独热处理）
    ohe = ColumnTransformer([('race', OneHotEncoder(), [0]), ('sex', OneHotEncoder(), [1])], remainder='passthrough')
    data = ohe.fit_transform(data[:,:])

    # 将数据集拆分为训练集和测试集
    #data_train, data_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=0)
    data_train, data_test, y_train, y_test = train_test_split(data, y, test_size=0.2)
    data = pd.DataFrame(data)
    data.columns = ['ra_Ame', 'ra_Asi', 'ra_Bla', 'ra_His', 'ra_Mex', 'ra_Oth', 'ra_Pue', 'ra_Whi', 'sex_1', 'sex_2',
                  'LSAT', 'UGPA']
    #训练集和测试集
    data_train = pd.DataFrame(data_train)
    data_test = pd.DataFrame(data_test)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    data_train.columns = ['ra_Ame', 'ra_Asi', 'ra_Bla', 'ra_His', 'ra_Mex', 'ra_Oth', 'ra_Pue', 'ra_Whi', 'sex_1', 'sex_2',
                    'LSAT', 'UGPA']
    data_test.columns = ['ra_Ame', 'ra_Asi', 'ra_Bla', 'ra_His', 'ra_Mex', 'ra_Oth', 'ra_Pue', 'ra_Whi', 'sex_1', 'sex_2',
                    'LSAT', 'UGPA']
    y_train.columns = ['ZFYA']
    y_test.columns = ['ZFYA']
    return data, data_train, data_test, y_train, y_test

#unfair full model
def unfair_full(data_train, data_test, y_train, y_test):
    mlRegressor = LinearRegression()
    #模型训练（使用所有属性）
    model = mlRegressor.fit(data_train, y_train)
    #print(mlRegressor.coef_) #训练后模型权重
    #模型预测
    y_pred= mlRegressor.predict(data_test)
    #模型评估(RMSE 均方根误差)
    sum_mean = 0
    for i in range(len(y_pred)):
        sum_mean+=(y_pred[i] - y_test.values[i])**2
    sum_erro = np.sqrt(sum_mean/len(y_test))
    return y_pred, sum_erro

#unfair unaware model
def unfair_unaware(data_train, data_test, y_train, y_test):
    mlRegressor = LinearRegression()
    # 模型训练(仅使用非保护属性)
    model = mlRegressor.fit(data_train.iloc[:,10:12], y_train)
    #print(mlRegressor.coef_) #训练后模型权重
    # 模型预测
    y_pred = mlRegressor.predict(data_test.iloc[:,10:12])
    # 模型评估(RMSE 均方根误差)
    sum_mean = 0
    for i in range(len(y_pred)):
        sum_mean += (y_pred[i] - y_test.values[i]) ** 2
    sum_erro = np.sqrt(sum_mean / len(y_test))
    return y_pred, sum_erro

#fair,deterministic model(Fair add)
def deterministic(data_train, data_test, y_train, y_test):
    mlRegressor_ugpa = LinearRegression()
    mlRegressor_lsat = LinearRegression()
    mlRegressor_det = LinearRegression()

    mlRegressor_ugpa_te = LinearRegression()
    mlRegressor_lsat_te = LinearRegression()

    # 模型训练
    #使用受保护属性拟合ugpa
    model_ugpa = mlRegressor_ugpa.fit(data_train.iloc[:,:10], data_train[['UGPA']])
    #ugpa_pred = mlRegressor_ugpa.predict(data_test.iloc[:,:10])
    #使用受保护属性拟合lsat
    model_lsat = mlRegressor_lsat.fit(data_train.iloc[:,:10], data_train[['LSAT']])
    #lsat_pred = mlRegressor_lsat.predict(data_test.iloc[:, :10])
    #Resid
    resid_UGPA_train = data_train[['UGPA']] - mlRegressor_ugpa.predict(data_train.iloc[:, 0:10])
    resid_LSAT_train = data_train[['LSAT']] - mlRegressor_lsat.predict(data_train.iloc[:, 0:10])

    con_ugls = pd.concat([resid_UGPA_train, resid_LSAT_train], axis = 1)
    model_det = mlRegressor_det.fit(con_ugls, y_train)#获得FYA模型

    # 模型测试
    # 使用受保护属性拟合ugpa得到测试eG
    model_ugpa_te = mlRegressor_ugpa_te.fit(data_test.iloc[:, :10], data_test[['UGPA']])
    # 使用受保护属性拟合lsat得到测试eL
    model_lsat_te = mlRegressor_lsat_te.fit(data_test.iloc[:, :10], data_test[['LSAT']])

    # Resid
    resid_UGPA_test = data_test[['UGPA']] - mlRegressor_ugpa_te.predict(data_test.iloc[:, 0:10])
    resid_LSAT_test = data_test[['LSAT']] - mlRegressor_lsat_te.predict(data_test.iloc[:, 0:10])
    con_ugls_te = pd.concat([resid_UGPA_test, resid_LSAT_test], axis=1) #测试集

    pre_det_te = mlRegressor_det.predict(con_ugls_te)

    sum_mean = 0
    for i in range(len(pre_det_te)):
        sum_mean += (pre_det_te[i] - y_test.values[i]) ** 2
    sum_erro = np.sqrt(sum_mean / len(y_test))
    return pre_det_te, sum_erro

#fair,nondeterministic model(Fair k)
def nondeterministic(data_train, data_test, y_train, y_test):
    #得到后验的训练集k(训练集)
    k_tr, ugpa0, k_u_ugpa, r_u_ugpa, sigma, lsat0, k_l_lsat, r_l_lsat = nondeterministic_tr(data_train,y_train) #得到训练集k

    # 参数
    ugpa0 = mean(ugpa0)
    k_u_ugpa = mean(k_u_ugpa)
    r_u_ugpa = np.mean(r_u_ugpa)
    sigma = mean(sigma)

    lsat0 = mean(lsat0)
    k_l_lsat = mean(k_l_lsat)
    r_l_lsat = np.mean(r_l_lsat)

    # 得到后验的测试集K（测试集）
    k_te = nondeterministic_te(data_test,ugpa0, k_u_ugpa, r_u_ugpa, sigma) #得到测试集k

    # 训练：用训练集k和因变量做线性拟合
    mlRegressor_fya = LinearRegression()
    model_fya = mlRegressor_fya.fit(k_tr, y_train)

    # 测试：用测试集K测试回归方差
    y_pred = mlRegressor_fya.predict(k_te)

    # 模型评估(RMSE 均方根误差)
    sum_mean = 0
    for i in range(len(y_pred)):
        sum_mean += (y_pred[i] - y_test.values[i]) ** 2
    sum_erro = np.sqrt(sum_mean / len(y_test))
    return y_pred, sum_erro

if __name__=="__main__":
    #获取预处理后的数据
    data, data_train, data_test, y_train, y_test = load_csv('law_data.csv')

    # unfair unaware model 预测值
    y_pred1, sum_erro1 = unfair_full(data_train, data_test, y_train, y_test)

    # unfair unaware model
    y_pred2, sum_erro2 = unfair_unaware(data_train, data_test, y_train, y_test)

    # fair,deterministic model(Fair add)
    y_pred3, sum_erro3 = deterministic(data_train, data_test, y_train, y_test)

    print(sum_erro1, sum_erro2, sum_erro3)

    y_pred4, sum_erro4 = nondeterministic(data_train, data_test, y_train, y_test)

    print(sum_erro4)
















