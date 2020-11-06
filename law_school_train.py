import pymc3 as pm#贝叶斯估计 MCMC拟合包

def nondeterministic_tr(data_train,y_train):
    with pm.Model() as ml:
        # 先验（待测变量）
        k = pm.Normal('k', mu=0, sd=1, shape=len(data_train))
        ugpa0 = pm.Normal('ugpa0', mu=0, sd=1)
        k_u_ugpa = pm.Normal('k_u_ugpa', mu=0, sd=1)
        r_u_ugpa = pm.Normal('r_u_ugpa', mu=0, sd=1, shape=10)
        sigma = pm.HalfNormal('sigma', sd=1)

        lsat0 = pm.Normal('lsat0', mu=0, sd=1)
        k_l_lsat = pm.Normal('k_l_lsat', mu=0, sd=1)
        r_l_lsat = pm.Normal('r_l_lsat', mu=0, sd=1, shape=10)

        k_f_fya = pm.Normal('k_f_fya', mu=0, sd=1)
        r_f_fya = pm.Normal('r_f_fya', mu=0, sd=1, shape=10)

        # 似然
        data_train['a'] = data_train[
            ['ra_Ame', 'ra_Asi', 'ra_Bla', 'ra_His', 'ra_Mex', 'ra_Oth', 'ra_Pue', 'ra_Whi', 'sex_1', 'sex_2']].apply(
            lambda x: x['ra_Ame'] * r_u_ugpa[0] + x['ra_Asi'] * r_u_ugpa[1] + x['ra_Bla'] * r_u_ugpa[2] + x['ra_His'] *
                      r_u_ugpa[3] + x['ra_Mex'] *
                      r_u_ugpa[4] + x['ra_Oth'] * r_u_ugpa[5] + x['ra_Pue'] * r_u_ugpa[6] + x['ra_Whi'] * r_u_ugpa[7] +
                      x['sex_1'] * r_u_ugpa[8] + x['sex_2'] * r_u_ugpa[9], axis=1)

        mu_ugpa = ugpa0 + k_u_ugpa * k + data_train['ra_Ame'].tolist()
        gpa = pm.Normal('ugpa', mu=mu_ugpa, sd=sigma, observed=data_train['UGPA'].tolist())

        data_train['b'] = data_train[
            ['ra_Ame', 'ra_Asi', 'ra_Bla', 'ra_His', 'ra_Mex', 'ra_Oth', 'ra_Pue', 'ra_Whi', 'sex_1', 'sex_2']].apply(
            lambda x: x['ra_Ame'] * r_l_lsat[0] + x['ra_Asi'] * r_l_lsat[1] + x['ra_Bla'] * r_l_lsat[2] + x['ra_His'] *
                      r_l_lsat[3] + x['ra_Mex'] *
                      r_l_lsat[4] + x['ra_Oth'] * r_l_lsat[5] + x['ra_Pue'] * r_l_lsat[6] + x['ra_Whi'] * r_l_lsat[7] +
                      x['sex_1'] * r_l_lsat[8] + x['sex_2'] * r_l_lsat[9], axis=1)

        mu_lsat = lsat0 + k_l_lsat * k + data_train['b'].tolist()
        sat = pm.Poisson('lsat', mu=mu_lsat, observed=data_train['LSAT'].tolist())

        data_train['c'] = data_train[
            ['ra_Ame', 'ra_Asi', 'ra_Bla', 'ra_His', 'ra_Mex', 'ra_Oth', 'ra_Pue', 'ra_Whi', 'sex_1', 'sex_2']].apply(
            lambda x: x['ra_Ame'] * r_f_fya[0] + x['ra_Asi'] * r_f_fya[1] + x['ra_Bla'] * r_f_fya[2] + x['ra_His'] *
                      r_f_fya[3] + x['ra_Mex'] *
                      r_f_fya[4] + x['ra_Oth'] * r_f_fya[5] + x['ra_Pue'] * r_f_fya[6] + x['ra_Whi'] * r_f_fya[7] +
                      x['sex_1'] * r_f_fya[8] + x['sex_2'] * r_f_fya[9], axis=1)
        mu_fya = k_f_fya * k + data_train['c'].tolist()
        fya = pm.Normal('zfya', mu=mu_fya, sd=1, observed = y_train['ZFYA'].tolist())

    with ml:
        # 估计后验（待测变量）
        trace = pm.sample(2000)
        print('k', trace['k'].shape)
        print(trace['k'][0:5], "\n")

    return k, ugpa0, k_u_ugpa, r_u_ugpa, sigma, lsat0, k_l_lsat, r_l_lsat