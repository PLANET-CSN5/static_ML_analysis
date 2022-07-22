from unittest import result
import pandas as pd
import numpy as np 
import datetime
import sklearn
import math
import sys

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

from matplotlib import pyplot

from pprint import pprint

##################################################################################33

def extract_given_prov (data, id):

    perprovincia = data[data["PROVINCIADOMICILIORESIDENZA"] == float(id)]
    #print(perprovincia.columns)

    properperprov = {}

    for col in perprovincia.columns:
        properperprov[col] = []

    for index, row in perprovincia.iterrows():

        for col in ["DATAPRELIEVO", "DATADIAGNOSI", \
            "DATAINIZIOSINTOMI", "DATADECESSO", "DATARICOVERO", \
            "DATATERAPIAINTENSIVA"]:

            date = row[col]
            date_time_obj = ""
            if type(date) == str:
                sdate = date.split("/")

                if len(sdate) == 3:
                    Y = sdate[2]
                    M = sdate[1]
                    D = sdate[0]
                    date_time_str = "%s-%s-%s"%(Y, M, D)

                    date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d')

            if date_time_obj == "":
                date_time_str = "%s-%s-%s"%("1975", "01", "01")
                date_time_obj = datetime.datetime.strptime(date_time_str, '%Y-%m-%d')
                
            properperprov[col].append(date_time_obj)
    
        for col in ['REGIONEDIAGNOSI', 'SESSO', 'ETA', 'PROVINCIADOMICILIORESIDENZA', \
            'NAZIONALITA', 'CASOIMPORTATO', 'SINTOMATICO', 'OPERATORESANITARIO', 'DECEDUTO',\
                'RICOVERO', 'TERAPIAINTENSIVA']:
            properperprov[col].append(row[col])

    newdata = pd.DataFrame.from_dict(properperprov)

    return newdata

##################################################################################33

def get_num_of_in_range(newdata, startdate, enddate, 
    entryname= "DATAPRELIEVO"):
    
    start_date = datetime.datetime.strptime(startdate, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(enddate, '%Y-%m-%d')
    mask = (newdata[entryname] >= start_date) & (newdata[entryname] <= end_date)
    selected = newdata.loc[mask]

    return selected.shape[0]

##################################################################################33

def extraxtnumber (indata, id, sdate, edate):
    newdata = extract_given_prov (indata, id)

    dataprelievo = get_num_of_in_range (newdata, str(sdate), str(edate), "DATAPRELIEVO")
    datasintomi = get_num_of_in_range (newdata, str(sdate), str(edate), "DATAINIZIOSINTOMI")
    datadiagnosi = get_num_of_in_range (newdata, str(sdate), str(edate), "DATADIAGNOSI")

    return dataprelievo, datasintomi, datadiagnosi

##################################################################################33

def rfregressors_custom_optimizer (Xin, yin, verbose=True, inboot=[True, False]):

    n_estimators = [100, 300, 500, 800, 1200]
    max_depth = [None, 5, 8, 15, 25, 30]
    min_samples_split = [2, 5, 10, 15, 100]
    min_samples_leaf = [1, 2, 5, 10] 
    random_state = [1]
    max_features = ['auto', 'sqrt']
    bootstrap = inboot

    hyperF = {"n_estimators" : n_estimators, 
            "max_depth" : max_depth,  
            "min_samples_split" : min_samples_split, 
            "min_samples_leaf" : min_samples_leaf, 
            "random_state" : random_state, 
            "bootstrap" : bootstrap,
            "max_features" : max_features}

    besthyperF = {"n_estimators" : n_estimators, 
            "max_depth" : max_depth,  
            "min_samples_split" : min_samples_split, 
            "min_samples_leaf" : min_samples_leaf, 
            "random_state" : random_state, 
            "bootstrap" : bootstrap,
            "max_features" : max_features}



    total = 1
    for k in hyperF:
        total *= len(hyperF[k])
    counter = 1
    bestmse = float("+inf")
    for a in hyperF["n_estimators"]:
        for b in  hyperF["max_depth"]:
            for c in  hyperF["min_samples_split"]:
                for d in  hyperF["min_samples_leaf"]:
                    for e in  hyperF["random_state"]:
                        for f in  hyperF["bootstrap"]:
                            for g in  hyperF["max_features"]:
                                model = RandomForestRegressor(
                                    n_estimators=a,
                                    max_depth=b,
                                    min_samples_split=c,
                                    min_samples_leaf=d,
                                    random_state=e,
                                    bootstrap=f,
                                    max_features=g
                                )
                                model.fit(Xin, yin)
 
                                y_pred = model.predict(Xin)
 
                                mse = sklearn.metrics.mean_squared_error(yin, y_pred)
                                #print(counter , " of ", total ,"MSE: ", mse)
                                counter += 1
                                if mse < bestmse:
                                    bestmse = mse

                                    besthyperF = {"n_estimators" : a,
                                                  "max_depth" : b,  
                                                  "min_samples_split" : c, 
                                                  "min_samples_leaf" : d, 
                                                  "random_state" : e, 
                                                  "bootstrap" : f,
                                                  "max_features" : g}




    return besthyperF, bestmse

##################################################################################33

def rfregressors_custom_optimizer_nooverfit (Xin, yin, verbose=True, inboot=[True, False]):

    n_estimators = [100, 300, 500, 800, 1200]
    max_depth = [None, 5, 8, 15, 25, 30]
    min_samples_split = [2, 5, 10, 15, 100]
    min_samples_leaf = [1, 2, 5, 10] 
    random_state = [1]
    max_features = ['auto', 'sqrt']
    bootstrap = inboot

    hyperF = {"n_estimators" : n_estimators, 
            "max_depth" : max_depth,  
            "min_samples_split" : min_samples_split, 
            "min_samples_leaf" : min_samples_leaf, 
            "random_state" : random_state, 
            "bootstrap" : bootstrap,
            "max_features" : max_features}

    besthyperF = {"n_estimators" : n_estimators, 
            "max_depth" : max_depth,  
            "min_samples_split" : min_samples_split, 
            "min_samples_leaf" : min_samples_leaf, 
            "random_state" : random_state, 
            "bootstrap" : bootstrap,
            "max_features" : max_features}
    
    X_train, X_test, y_train, y_test = train_test_split(
            Xin, yin, test_size=0.35)

    total = 1
    for k in hyperF:
        total *= len(hyperF[k])
    counter = 1
    best_train_rmse = float("+inf")
    best_test_rmse = float("+inf")
    best_diff = float("+inf")
    for a in hyperF["n_estimators"]:
        for b in  hyperF["max_depth"]:
            for c in  hyperF["min_samples_split"]:
                for d in  hyperF["min_samples_leaf"]:
                    for e in  hyperF["random_state"]:
                        for f in  hyperF["bootstrap"]:
                            for g in  hyperF["max_features"]:
                                model = RandomForestRegressor(
                                    n_estimators=a,
                                    max_depth=b,
                                    min_samples_split=c,
                                    min_samples_leaf=d,
                                    random_state=e,
                                    bootstrap=f,
                                    max_features=g
                                ]

                                model.fit(X_train, y_train)

                                y_pred = model.predict(X_train)
                                mse = sklearn.metrics.mean_squared_error(y_train, y_pred)
                                train_rmse = math.sqrt(mse)
                                
                                y_pred = model.predict(X_test)
                                mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
                                test_rmse = math.sqrt(mse)
 
                                diffrmse = math.fabs(test_rmse -train_rmse)

                                percdiff = diffrmse/((test_rmse + train_rmse)/2.0)

                                #print(counter , " of ", total ,"Train RMSE: ", train_rmse)
                                #print(counter , " of ", total ," Test RMSE: ", test_rmse)
                                counter += 1
                                if percdiff <= 0.3 and train_rmse < best_train_rmse and test_rmse < best_test_rmse and \
                                    diffrmse < best_diff:
                                    best_test_rmse = test_rmse
                                    best_train_rmse = train_rmse
                                    best_diff = diffrmse

                                    besthyperF = {"n_estimators" : a,
                                                  "max_depth" : b,  
                                                  "min_samples_split" : c, 
                                                  "min_samples_leaf" : d, 
                                                  "random_state" : e, 
                                                  "bootstrap" : f,
                                                  "max_features" : g}

    return besthyperF, best_diff, best_test_rmse, best_train_rmse

##################################################################################33

def rfregressors_custom_optimizer_testset (Xin, yin, verbose=True, inboot=[True, False]):

    n_estimators = [100, 300, 500, 800, 1200]
    max_depth = [None, 5, 8, 15, 25, 30]
    min_samples_split = [2, 5, 10, 15, 100]
    min_samples_leaf = [1, 2, 5, 10] 
    random_state = [1]
    max_features = ['auto', 'sqrt']
    bootstrap = inboot

    hyperF = {"n_estimators" : n_estimators, 
            "max_depth" : max_depth,  
            "min_samples_split" : min_samples_split, 
            "min_samples_leaf" : min_samples_leaf, 
            "random_state" : random_state, 
            "bootstrap" : bootstrap,
            "max_features" : max_features}

    besthyperF = {"n_estimators" : n_estimators, 
            "max_depth" : max_depth,  
            "min_samples_split" : min_samples_split, 
            "min_samples_leaf" : min_samples_leaf, 
            "random_state" : random_state, 
            "bootstrap" : bootstrap,
            "max_features" : max_features}
    
    X_train, X_test, y_train, y_test = train_test_split(
            Xin, yin, test_size=0.35)

    total = 1
    for k in hyperF:
        total *= len(hyperF[k])
    counter = 1
    best_train_rmse = float("+inf")
    best_test_rmse = float("+inf")
    best_diff = float("+inf")
    for a in hyperF["n_estimators"]:
        for b in  hyperF["max_depth"]:
            for c in  hyperF["min_samples_split"]:
                for d in  hyperF["min_samples_leaf"]:
                    for e in  hyperF["random_state"]:
                        for f in  hyperF["bootstrap"]:
                            for g in  hyperF["max_features"]:
                                model = RandomForestRegressor(
                                    n_estimators=a,
                                    max_depth=b,
                                    min_samples_split=c,
                                    min_samples_leaf=d,
                                    random_state=e,
                                    bootstrap=f,
                                    max_features=g
                                ]

                                model.fit(X_train, y_train)

                                y_pred = model.predict(X_train)
                                mse = sklearn.metrics.mean_squared_error(y_train, y_pred)
                                train_rmse = math.sqrt(mse)
                                
                                y_pred = model.predict(X_test)
                                mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
                                test_rmse = math.sqrt(mse)
 
                                diffrmse = math.fabs(test_rmse -train_rmse)

                                percdiff = diffrmse/((test_rmse + train_rmse)/2.0)

                                #print(counter , " of ", total ,"Train RMSE: ", train_rmse)
                                #print(counter , " of ", total ," Test RMSE: ", test_rmse)
                                counter += 1
                                if test_rmse < best_test_rmse:
                                    best_test_rmse = test_rmse
                                    best_train_rmse = train_rmse
                                    best_diff = diffrmse

                                    besthyperF = {"n_estimators" : a,
                                                  "max_depth" : b,  
                                                  "min_samples_split" : c, 
                                                  "min_samples_leaf" : d, 
                                                  "random_state" : e, 
                                                  "bootstrap" : f,
                                                  "max_features" : g}

    return besthyperF, best_diff, best_test_rmse, best_train_rmse

##################################################################################33

def rfregressors_optimizer (Xin, yin, verbose=True):

    n_estimators = [100, 300, 500, 800, 1200]
    max_depth = [None, 5, 8, 15, 25, 30]
    min_samples_split = [2, 5, 10, 15, 100]
    min_samples_leaf = [1, 2, 5, 10] 
    random_state = [1]
    max_features = ['auto', 'sqrt']
    bootstrap = [True, False]

    hyperF = {"n_estimators" : n_estimators, 
            "max_depth" : max_depth,  
            "min_samples_split" : min_samples_split, 
            "min_samples_leaf" : min_samples_leaf, 
            "random_state" : random_state, 
            "bootstrap" : bootstrap,
            "max_features" : max_features}
 
    model = RandomForestRegressor()

    gridF = GridSearchCV(model, hyperF, cv = 3, verbose = 1, 
                      n_jobs = -1)

    model.fit(Xin, yin)
    bestF = gridF.fit(Xin, yin)

    if verbose:
        pprint(bestF.best_params_)

    return bestF

##################################################################################33

def dropcol_importances(rf, X_train, y_train):
    from sklearn.base import clone

    rf_ = clone(rf)
    rf_.random_state = 999
    rf_.fit(X_train, y_train)
    baseline = rf_.oob_score_
    imp = []
    for col in X_train.columns:
        X = X_train.drop(col, axis=1)
        rf_ = clone(rf)
        rf_.random_state = 999
        rf_.fit(X, y_train)
        o = rf_.oob_score_
        imp.append(baseline - o)
    imp = np.array(imp)
    I = pd.DataFrame(
            data={'Feature':X_train.columns,
                  'Importance':imp})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=True)
    
    return I

##################################################################################33

def rfregressors (Xin, yin, features, plotname="rf_model", N = 50, verbose=True,
    pout=sys.stdout, showplot=False, optimisedparams=None ):

    train_rmse = []
    test_rmse = []
 
    for isplit in range(N):
        X_train, X_test, y_train, y_test = train_test_split(
            Xin, yin, test_size=0.35)
        model = None 

        if optimisedparams is not None:
            model = RandomForestRegressor(**optimisedparams)
        else:
            model = RandomForestRegressor(random_state = 1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_train)
        mse = sklearn.metrics.mean_squared_error(y_train, y_pred)
        rmse = math.sqrt(mse)
        train_rmse.append(rmse)
        
        y_pred = model.predict(X_test)
        mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
        rmse = math.sqrt(mse)
        test_rmse.append(rmse)

    trainavgrmse = (np.average(train_rmse), np.std(train_rmse))
    testavgrmse = (np.average(test_rmse), np.std(test_rmse)) 

    if verbose:     
        print("Training set average RMSE: %8.5f %8.5f "%(trainavgrmse[0], trainavgrmse[1]), 
            file=pout)
        print("    Test set average RMSE: %8.5f %8.5f "%(testavgrmse[0], testavgrmse[1]),
            file=pout)
      
    model = None

    if optimisedparams is not None:
        model = RandomForestRegressor(**optimisedparams)
    else:
        model = RandomForestRegressor(random_state = 1)

    if verbose:
        print("Parameters used: ")
        pprint(model.get_params())
 
    # fit the model
    model.fit(Xin, yin)
 
    y_pred = model.predict(Xin)
 
    mse = sklearn.metrics.mean_squared_error(yin, y_pred)
    r2s = sklearn.metrics.r2_score(yin, y_pred)

    rmse = math.sqrt(mse)
    if verbose:
        print("             Fullset RMSE: %8.5f"%rmse, file=pout)
        print("                       R2: %8.5f"%r2s, file=pout)
    
    fullsetrmse = rmse
 
    if verbose:
        pyplot.rcParams.update({'font.size': 22})
        pyplot.title("ScatterPlot predicted vs True")
        pyplot.scatter(yin, y_pred)

        if showplot:
            fig1 = pyplot.gcf()
            pyplot.figure(figsize=(10,10))
            pyplot.show()
            fig1.savefig(plotname+"_scatter.png", bbox_inches="tight")
        else:
            pyplot.savefig(plotname+"_scatter.png")
 
    # get importance
    importance = model.feature_importances_
    if verbose:
        print("",file=pout)
        print("Features importance from model: ",file=pout)
        # summarize feature importance
        for i,v in enumerate(importance):
            print('Feature: %30s, Score: %.5f' % (features[i],v),file=pout)
 
        # plot feature importance
        pyplot.clf()
        pyplot.figure(figsize=(10,10))
        pyplot.title("Features importance from model")
        pyplot.barh(features, importance)
        pyplot.xticks(rotation=45, ha="right")
        pyplot.gcf().subplots_adjust(bottom=0.30)
        if showplot:
            fig1 = pyplot.gcf()
            pyplot.figure(figsize=(10,10))
            pyplot.show()
            fig1.savefig(plotname+"_feats_imp_frommodel.png", bbox_inches="tight")
        else:
            pyplot.savefig(plotname+"_feats_imp_frommodel.png")
 
    #Permutation feature importance is a model inspection technique that 
    # can be used for any fitted estimator when the data is tabular. This 
    # is especially useful for non-linear or opaque estimators. The permutation 
    # feature importance is defined to be the decrease in a model score when a single 
    # feature value is randomly shuffled. This procedure breaks the relationship between 
    # the feature and the target, thus the drop in the model score is indicative of how 
    # much the model depends on the feature. This technique benefits from being model 
    # agnostic and can be calculated many times with different permutations of the feature.
 
    # When two features are correlated and one of the features is permuted, the model 
    # will still have access to the feature through its correlated feature.
 
    model = RandomForestRegressor()
 
    # fit the model
    model.fit(Xin, yin)
    # perform permutation importance
    #scoringset = ['r2', 'neg_mean_squared_error']
    #results = {}

    #for s in scoringset:
    #    results[s] = None
    #    results[s] = permutation_importance(model, Xin, yin, n_repeats=50, random_state=0, \
    #        scoring=s)
    # get importance
    results= permutation_importance(model, Xin, yin, n_repeats=50, random_state=0, \
        scoring="neg_mean_squared_error")

    importance = results.importances_mean
    importanceerror = results.importances_std
    # summarize feature importance

    if verbose:
        print("",file=pout)
        print("Features importance from Permutation Score neg_mean_squared_error : ",file=pout)

    totfi = 0.0
    featimport = {}
    for i,v in enumerate(importance):
        featimport[features[i]] = v
        if verbose:
            print('Feature: %30s, Score: %.5f +/- %.5f' % (features[i],v,importanceerror[i]),file=pout)
        totfi += v

    for i,v in enumerate(importance):
        featimport[features[i]] /= totfi

    if verbose:
        # plot feature importance
        pyplot.clf()
        pyplot.figure(figsize=(10,10))
        pyplot.title("Features importance from Permutation [neg_mean_squared_error]")
        pyplot.barh(features, importance, xerr=importanceerror, capsize=10)
        pyplot.xticks(rotation=45, ha="right")
        pyplot.gcf().subplots_adjust(bottom=0.30)
        if showplot:
            fig1 = pyplot.gcf()
            pyplot.figure(figsize=(10,10))
            pyplot.show()
            fig1.savefig(plotname+"_feats_imp_frompermutation_neg_mean_squared_error.png", bbox_inches="tight")
        else:
            pyplot.savefig(plotname+"_feats_imp_frompermutation_neg_mean_squared_error.png")

    results= permutation_importance(model, Xin, yin, n_repeats=50, random_state=0, \
        scoring="r2")

    importance = results.importances_mean
    importanceerror = results.importances_std
    # summarize feature importance

    if verbose:
        print("",file=pout)
        print("Features importance from Permutation Score r2: ",file=pout)

    totfi = 0.0
    featimport2 = {}
    for i,v in enumerate(importance):
        featimport2[features[i]] = v
        if verbose:
            print('Feature: %30s, Score: %.5f +/- %.5f' % (features[i],v,importanceerror[i]),file=pout)
        totfi += v

    for i,v in enumerate(importance):
        featimport2[features[i]] /= totfi

    if verbose:
        # plot feature importance
        pyplot.clf()
        pyplot.figure(figsize=(10,10))
        pyplot.title("Features importance from Permutation [r2]")
        pyplot.barh(features, importance, xerr=importanceerror, capsize=10)
        pyplot.xticks(rotation=45, ha="right")
        pyplot.gcf().subplots_adjust(bottom=0.30)
        if showplot:
            fig1 = pyplot.gcf()
            pyplot.figure(figsize=(10,10))
            pyplot.show()
            fig1.savefig(plotname+"_feats_imp_frompermutation_r2.png", bbox_inches="tight")
        else:
            pyplot.savefig(plotname+"_feats_imp_frompermutation_r2.png")

    
    #test in the validation set
    X_train, X_test, y_train, y_test = train_test_split(Xin, yin, test_size = 0.2, random_state = 42)
    model = RandomForestRegressor()
 
    # fit the model
    model.fit(X_train, y_train)

    results= permutation_importance(model, X_test, y_test, n_repeats=50, random_state=0, \
        scoring="neg_mean_squared_error")

    importance = results.importances_mean
    importanceerror = results.importances_std
    # summarize feature importance

    if verbose:
        print("",file=pout)
        print("Features importance from Permutation TetSet Score neg_mean_squared_error : ",file=pout)

    totfi = 0.0
    featimport3 = {}
    for i,v in enumerate(importance):
        featimport3[features[i]] = v
        if verbose:
            print('Feature: %30s, Score: %.5f +/- %.5f' % (features[i],v,importanceerror[i]),file=pout)
        totfi += v

    for i,v in enumerate(importance):
        featimport3[features[i]] /= totfi

    if verbose:
        # plot feature importance
        pyplot.clf()
        pyplot.figure(figsize=(10,10))
        pyplot.title("Features importance from Permutation TestSet [neg_mean_squared_error]")
        pyplot.barh(features, importance, xerr=importanceerror, capsize=10)
        pyplot.xticks(rotation=45, ha="right")
        pyplot.gcf().subplots_adjust(bottom=0.30)
        if showplot:
            fig1 = pyplot.gcf()
            pyplot.figure(figsize=(10,10))
            pyplot.show()
            fig1.savefig(plotname+"_feats_imp_frompermutation_testset_neg_mean_squared_error.png", bbox_inches="tight")
        else:
            pyplot.savefig(plotname+"_feats_imp_frompermutation_testset_neg_mean_squared_error.png")

    results= permutation_importance(model, X_test, y_test, n_repeats=50, random_state=0, \
        scoring="r2")

    importance = results.importances_mean
    importanceerror = results.importances_std
    # summarize feature importance

    if verbose:
        print("",file=pout)
        print("Features importance from Permutation TestSet Score r2: ",file=pout)

    totfi = 0.0
    featimport4 = {}
    for i,v in enumerate(importance):
        featimport4[features[i]] = v
        if verbose:
            print('Feature: %30s, Score: %.5f +/- %.5f' % (features[i],v,importanceerror[i]),file=pout)
        totfi += v

    for i,v in enumerate(importance):
        featimport4[features[i]] /= totfi

    if verbose:
        # plot feature importance
        pyplot.clf()
        pyplot.figure(figsize=(10,10))
        pyplot.title("Features importance from Permutation TestSet [r2]")
        pyplot.barh(features, importance, xerr=importanceerror, capsize=10)
        pyplot.xticks(rotation=45, ha="right")
        pyplot.gcf().subplots_adjust(bottom=0.30)
        if showplot:
            fig1 = pyplot.gcf()
            pyplot.figure(figsize=(10,10))
            pyplot.show()
            fig1.savefig(plotname+"_feats_imp_frompermutation_testset_r2.png", bbox_inches="tight")
        else:
            pyplot.savefig(plotname+"_feats_imp_frompermutation_testset_r2.png")

    return trainavgrmse, testavgrmse, fullsetrmse, featimport, featimport2,  featimport3, featimport4

##################################################################################33


def knregressors (Xin, yin, features, plotname="KNmodel", N=50, verbose=True):
    train_rmse = []
    test_rmse = []
 
    # The entire training dataset is stored. When a prediction is 
    # required, the k-most similar records to a new record from the 
    # training dataset are then located. From these neighbors, a 
    # summarized prediction is made.
 
    # Similarity between records can be measured many different ways. 
    # A problem or data-specific method can be used. Generally, with 
    # tabular data, a good starting point is the Euclidean distance.
 
    for isplit in range(N):
        X_train, X_test, y_train, y_test = train_test_split(
            Xin, yin, test_size=0.35)
        model = KNeighborsRegressor()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_train)
        mse = sklearn.metrics.mean_squared_error(y_train, y_pred)
        rmse = math.sqrt(mse)
        train_rmse.append(rmse)
        
        y_pred = model.predict(X_test)
        mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
        rmse = math.sqrt(mse)
        test_rmse.append(rmse)

    trainavgrmse = (np.average(train_rmse), np.std(train_rmse))
    testavgrmse = (np.average(test_rmse), np.std(test_rmse)) 

    if verbose:      
        print("Training set average RMSE: ", trainavgrmse[0], trainavgrmse[1])
        print("    Test set average RMSE: ", testavgrmse[0], testavgrmse[1])
        print(" ")
      
    model = KNeighborsRegressor()
    
    # fit the model
    model.fit(Xin, yin)
 
    y_pred = model.predict(Xin)
 
    mse = sklearn.metrics.mean_squared_error(yin, y_pred)
    rmse = math.sqrt(mse)
    y_pred = model.predict(Xin)

    fullsetrmse = rmse

    if verbose:
        print("Fullset RMSE: ", rmse)
        pyplot.title ("ScatterPlot Predicted vs True)")
        pyplot.scatter(yin, y_pred)
        #pyplot.show()
        pyplot.savefig(plotname+"_scatter.png")
      
    model = KNeighborsRegressor()
 
    # fit the model
    model.fit(Xin, yin)
    # perform permutation importance
    results = permutation_importance(model, Xin, yin, scoring='neg_mean_squared_error')
    # get importance
    importance = results.importances_mean  
    # summarize feature importance
    
    if verbose:
        print("")
        print("Features importance from Permutation: ")
        for i,v in enumerate(importance):	
            print('Feature: %s, Score: %.5f' % (features[i],v))
        # plot feature importance
        pyplot.bar(features, importance)
        #pyplot.show()
        pyplot.savefig(plotname+"_feats_imp_frompermutation.png")

    featimport = {}
    totfi = 0.0
    for i,v in enumerate(importance):
        featimport[features[i]] = v
        totfi += v
        
    for i,v in enumerate(importance):
        featimport[features[i]] /= totfi
        
    return trainavgrmse, testavgrmse, fullsetrmse, featimport

##################################################################################33

def printcsv (fullfeatset, features, rf, kn):

    kstr = ""
    for f in features:
        kstr += f + "_"
    print(kstr + "RF , ", end = "")
    print("%10.5f , %10.5f , %10.5f , %10.5f , %10.5f , "% \
          (rf[0][0], rf[0][1], rf[1][0], rf[1][1], rf[2]), end="")
    for k in fullfeatset:
        if k in rf[3]:
            print("%10.5f , "%(rf[3][k]), end="")
        else:
            print("0.0 , ", end="")
    print()
    print(kstr + "KN , ", end = "")
    print("%10.5f , %10.5f , %10.5f , %10.5f , %10.5f , "% \
          (kn[0][0], kn[0][1], kn[1][0], kn[1][1], kn[2]), end="")
    for i, k in enumerate(fullfeatset):
        if k in kn[3]:
            if i == len(fullfeatset) - 1:
                print("%10.5f "%(kn[3][k]))
            else:
                print("%10.5f , "%(kn[3][k]), end="")
        else:
            if i == len(fullfeatset) - 1:
                print("0.0 ")
            else:
                print("0.0 , ", end="")

##################################################################################33

def printcsvRF (fullfeatset, features, rf):

    kstr = ""
    for f in features:
        kstr += f + "_"
    print(kstr + "RF , ", file=sys.stderr, end = "")
    print("%10.5f , %10.5f , %10.5f , %10.5f , %10.5f , "% \
          (rf[0][0], rf[0][1], rf[1][0], rf[1][1], rf[2]), file=sys.stderr, end="")

    featvals = {}
    for i, k in enumerate(fullfeatset):
        if k in rf[3]:
            print("%10.5f , "%(rf[3][k]), file=sys.stderr, end="")
            featvals[k] = rf[3][k]
        else:
            print("0.0 , ", file=sys.stderr, end="")

    featsorted = {k: v for k, v in sorted(featvals.items(), \
        key=lambda item: item[1])}
    
    print (list(featsorted.items())[-1][0], " ", list(featsorted.items())[-2][0], \
        file=sys.stderr )

    return list(featsorted.items())[-1], list(featsorted.items())[-2]

##################################################################################33
