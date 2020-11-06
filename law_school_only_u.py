import pymc3 as pm

def nondeterministic_te(data_test,ugpa0,k_u_ugpa,r_u_ugpa,sigma,lsat0,k_l_lsat,r_l_lsat):
    with pm.Model() as ugpa_te:
        k = pm.Normal('k', mu=0, sd=1, shape=len(data_test))#先验
        data_test['a'] = data_test[['ra_Ame', 'ra_Asi', 'ra_Bla', 'ra_His', 'ra_Mex', 'ra_Oth', 'ra_Pue', 'ra_Whi', 'sex_1', 'sex_2']].apply(
            lambda x:  x['ra_Ame'] * r_u_ugpa[0] + x['ra_Asi'] * r_u_ugpa[1] + x['ra_Bla'] * r_u_ugpa[2] + x['ra_His'] * r_u_ugpa[3] + x['ra_Mex'] *
                      r_u_ugpa[4] + x['ra_Oth'] * r_u_ugpa[5] + x['ra_Pue'] * r_u_ugpa[6] + x['ra_Whi'] * r_u_ugpa[7] +
                      x['sex_1'] * r_u_ugpa[8] + x['sex_2'] * r_u_ugpa[9],axis=1)

        mu_ugpa = ugpa0 + k_u_ugpa * k + data_test['ra_Ame'].tolist()
        ugpa = pm.Normal('ugpa', mu = mu_ugpa, sd = sigma, observed=data_test['UGPA'].tolist())


        data_test['b'] = data_test[['ra_Ame', 'ra_Asi', 'ra_Bla', 'ra_His', 'ra_Mex', 'ra_Oth', 'ra_Pue', 'ra_Whi', 'sex_1', 'sex_2']].apply(
                lambda x: x['ra_Ame'] * r_l_lsat[0] + x['ra_Asi'] * r_l_lsat[1] + x['ra_Bla'] * r_l_lsat[2] + x['ra_His'] *r_l_lsat[3] + x['ra_Mex'] *
                          r_l_lsat[4] + x['ra_Oth'] * r_l_lsat[5] + x['ra_Pue'] * r_l_lsat[6] + x['ra_Whi'] * r_l_lsat[7] +
                          x['sex_1'] * r_l_lsat[8] + x['sex_2'] * r_l_lsat[9], axis=1)

        mu_lsat = lsat0 + k_l_lsat * k + data_test['b'].tolist()
        lsat = pm.Poisson('lsat', mu=mu_lsat, observed=data_test['LSAT'].tolist())

        with ugpa_te:
            trace = pm.sample(2000)
            print('k', trace['k'].shape)
            print(trace['k'][0:5], "\n")

    return k