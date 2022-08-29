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

    # quick check for testing purpose
    #n_estimators = [100, 300]
    #max_depth = [None, 5]
    #min_samples_split = [2, 5]
    #min_samples_leaf = [1, 2] 
    #random_state = [1]
    #max_features = ['auto']
    #bootstrap = inboot

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

                                diffstdperc = 100*(math.fabs(np.std(y_pred) - np.std(yin))/np.std(yin))
 
                                mse = sklearn.metrics.mean_squared_error(yin, y_pred)
                                #print(counter , " of ", total ,"MSE: ", mse)
                                counter += 1
                                if diffstdperc < 80.0 and mse < bestmse:
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
                                )

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

                                model1 = RandomForestRegressor(
                                    n_estimators=a,
                                    max_depth=b,
                                    min_samples_split=c,
                                    min_samples_leaf=d,
                                    random_state=e,
                                    bootstrap=f,
                                    max_features=g
                                )

                                model1.fit(Xin, yin)
                                
                                y_pred = model1.predict(Xin)
                                
                                diffstdperc = 100*(math.fabs(np.std(y_pred) - np.std(yin))/np.std(yin))

                                #print(diffstdperc, percdiff, diffrmse, train_rmse, test_rmse)
                                #print(best_train_rmse, best_test_rmse, best_diff)
                                #print(diffstdperc < 80.0)
                                #print(percdiff <= 0.9)
                                #print(train_rmse < best_train_rmse)
                                #print(test_rmse < best_test_rmse)
                                #print( diffrmse < best_diff)

                                counter += 1
                                if diffstdperc < 80.0 and percdiff <= 1.0 and train_rmse < best_train_rmse \
                                    and test_rmse < best_test_rmse and diffrmse < best_diff:

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

    # quick check for testing purpose
    #n_estimators = [100, 300]
    #max_depth = [None, 5]
    #min_samples_split = [2, 5]
    #min_samples_leaf = [1, 2] 
    #random_state = [1]
    #max_features = ['auto']
    #bootstrap = inboot

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
                                )

                                model.fit(X_train, y_train)

                                y_pred = model.predict(X_train)
                                mse = sklearn.metrics.mean_squared_error(y_train, y_pred)
                                train_rmse = math.sqrt(mse)
                                
                                y_pred = model.predict(X_test)
                                mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
                                test_rmse = math.sqrt(mse)
 
                                diffrmse = math.fabs(test_rmse -train_rmse)

                                model1 = RandomForestRegressor(
                                    n_estimators=a,
                                    max_depth=b,
                                    min_samples_split=c,
                                    min_samples_leaf=d,
                                    random_state=e,
                                    bootstrap=f,
                                    max_features=g
                                )

                                model1.fit(Xin, yin)
                                
                                y_pred = model1.predict(Xin)
                                
                                diffstdperc = 100*(math.fabs(np.std(y_pred) - np.std(yin))/np.std(yin))

                                #print(counter , " of ", total ,"Train RMSE: ", train_rmse)
                                #print(counter , " of ", total ," Test RMSE: ", test_rmse)
                                counter += 1
                                if diffstdperc < 80.0 and \
                                    test_rmse < best_test_rmse:

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

def rfregressors_custom_optimizer_trainset (Xin, yin, verbose=True, inboot=[True, False]):

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
                                )

                                model.fit(X_train, y_train)

                                y_pred = model.predict(X_train)
                                mse = sklearn.metrics.mean_squared_error(y_train, y_pred)
                                train_rmse = math.sqrt(mse)
                                
                                y_pred = model.predict(X_test)
                                mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
                                test_rmse = math.sqrt(mse)
 
                                diffrmse = math.fabs(test_rmse -train_rmse)

                                model1 = RandomForestRegressor(
                                    n_estimators=a,
                                    max_depth=b,
                                    min_samples_split=c,
                                    min_samples_leaf=d,
                                    random_state=e,
                                    bootstrap=f,
                                    max_features=g
                                )

                                model1.fit(Xin, yin)
                                
                                y_pred = model1.predict(Xin)
                                
                                diffstdperc = 100*(math.fabs(np.std(y_pred) - np.std(yin))/np.std(yin))

                                #print(counter , " of ", total ,"Train RMSE: ", train_rmse)
                                #print(counter , " of ", total ," Test RMSE: ", test_rmse)
                                counter += 1
                                if diffstdperc < 80.0 and \
                                    train_rmse < best_train_rmse:

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
    pout=sys.stdout, showplot=False, optimisedparams=None , alsofrommodel=False,
    visualmap=None):

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
     
    model = None

    if optimisedparams is not None:
        model = RandomForestRegressor(**optimisedparams)
    else:
        model = RandomForestRegressor(random_state = 1)
    
    model.fit(Xin, yin)

    if verbose:
        print("Parameters used: ")
        pprint(model.get_params())
        print("")

    if verbose:     
        print("Training set average RMSE: %8.5f %8.5f "%(trainavgrmse[0], trainavgrmse[1]), 
            file=pout)
        print("    Test set average RMSE: %8.5f %8.5f "%(testavgrmse[0], testavgrmse[1]),
            file=pout)

    y_pred = model.predict(Xin)

    print("")

    diffstdperc = 100*(math.fabs(np.std(y_pred) - np.std(yin))/np.std(yin))
    print("Prediction STD : %10.5f"%(np.std(y_pred)))
    print("True value STD : %10.5f"%(np.std(yin)))
    print("Difference in percentage: %10.5f"%(diffstdperc))

    print("")

    mse = sklearn.metrics.mean_squared_error(yin, y_pred)
    r2s = sklearn.metrics.r2_score(yin, y_pred)

    rmse = math.sqrt(mse)
    if verbose:
        print("             Fullset RMSE: %10.5f"%rmse, file=pout)
        print("                       R2: %10.5f"%r2s, file=pout)
    
    fullsetrmse = rmse
 
    if verbose:
        pyplot.rcParams.update({'font.size': 15})
        pyplot.title("Scatterplot for Full Set")
        pyplot.xlabel("log(Cases/Population)")
        pyplot.ylabel("Predicted log(Cases/Population)")
        pyplot.scatter(yin, y_pred)

        if showplot:
            fig1 = pyplot.gcf()
            pyplot.figure(figsize=(10,10))
            pyplot.show()
            fig1.savefig(plotname+"_fullset_scatter.png", bbox_inches="tight")
        else:
            pyplot.savefig(plotname+"_fullset_scatter.png")
 
    # get importance
    if alsofrommodel:
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
 
    if optimisedparams is not None:
        model = RandomForestRegressor(**optimisedparams)
    else:
        model = RandomForestRegressor(random_state = 1)
 
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
    results = permutation_importance(model, Xin, yin, n_repeats=50, random_state=0, \
        scoring="neg_mean_squared_error")

    importance = results.importances_mean
    importanceerror = results.importances_std
    # summarize feature importance

    featuresforplot = []

    if visualmap == None:
        featuresforplot = features
    else:
        for f in features:
          vname = visualmap[f]
          featuresforplot.append(vname)


    if verbose:
        print("",file=pout)
        print("Features importance from Permutation Fullset Score neg_mean_squared_error : ",file=pout)

    refvalue = 0.0
    for i,v in enumerate(importance):
        if features[i] == "randomfeature":
            refvalue = v

    totfi = 0.0
    featimport = {}
    for i,v in enumerate(importance):
        featimport[features[i]] = v
        if verbose:
            if (v > 0.0 and v > refvalue):
                print('Feature: %30s, Score: %.5f +/- %.5f' % \
                    (featuresforplot[i],v,importanceerror[i]),file=pout)
        totfi += v

    for i,v in enumerate(importance):
        featimport[features[i]] /= totfi


    if verbose:
        # plot feature importance
        pyplot.clf()
        pyplot.figure(figsize=(10,10))
        pyplot.title("Features importance for the Full Set [Neg. MSE]")
        pyplot.barh(featuresforplot, importance, xerr=importanceerror, capsize=10)
        pyplot.xticks(rotation=45, ha="right")
        pyplot.gcf().subplots_adjust(bottom=0.30)
        if showplot:
            fig1 = pyplot.gcf()
            pyplot.figure(figsize=(10,10))
            pyplot.show()
            fig1.savefig(plotname+"_fullset_feats_imp_frompermutation_neg_mean_squared_error.png", bbox_inches="tight")
        else:
            pyplot.savefig(plotname+"_fullset_feats_imp_frompermutation_neg_mean_squared_error.png")

    results= permutation_importance(model, Xin, yin, n_repeats=50, random_state=0, \
        scoring="r2")

    importance = results.importances_mean
    importanceerror = results.importances_std
    # summarize feature importance

    if verbose:
        print("",file=pout)
        print("Features importance from Permutation Full Set Score r2: ",file=pout)

    refvalue = 0.0
    for i,v in enumerate(importance):
        if features[i] == "randomfeature":
            refvalue = v

    totfi = 0.0
    featimport2 = {}
    for i,v in enumerate(importance):
        featimport2[features[i]] = v
        if verbose:
            if (v > 0.0 and v > refvalue):
                print('Feature: %30s, Score: %.5f +/- %.5f' % (\
                    featuresforplot[i],v,importanceerror[i]),file=pout)
        totfi += v

    for i,v in enumerate(importance):
        featimport2[features[i]] /= totfi

    if verbose:
        # plot feature importance
        pyplot.clf()
        pyplot.figure(figsize=(10,10))
        pyplot.title("Features importance for the Full Set [R2]")
        pyplot.barh(featuresforplot, importance, xerr=importanceerror, capsize=10)
        pyplot.xticks(rotation=45, ha="right")
        pyplot.gcf().subplots_adjust(bottom=0.30)
        if showplot:
            fig1 = pyplot.gcf()
            pyplot.figure(figsize=(10,10))
            pyplot.show()
            fig1.savefig(plotname+"_fullset_feats_imp_frompermutation_r2.png", bbox_inches="tight")
        else:
            pyplot.savefig(plotname+"_fullset_feats_imp_frompermutation_r2.png")
    
    #test in the validation set
    X_train, X_test, y_train, y_test = train_test_split(Xin, yin, test_size = 0.2, random_state = 42)
 
    # fit the model
    if optimisedparams is not None:
        model = RandomForestRegressor(**optimisedparams)
    else:
        model = RandomForestRegressor(random_state = 1)
 
    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)

    mse = sklearn.metrics.mean_squared_error(y_train, y_pred)
    r2s = sklearn.metrics.r2_score(y_train, y_pred)

    rmse = math.sqrt(mse)
    if verbose:
        print("         Trainingset RMSE: %10.5f"%rmse, file=pout)
        print("                       R2: %10.5f"%r2s, file=pout)

    if verbose:
        pyplot.clf()
        pyplot.figure(figsize=(10,10))
        pyplot.rcParams.update({'font.size': 15})
        pyplot.title("Scatterplot for Training Set")
        pyplot.xlabel("log(Cases/Population)")
        pyplot.ylabel("Predicted log(Cases/Population)")
        pyplot.scatter(y_train, y_pred)

        if showplot:
            fig1 = pyplot.gcf()
            pyplot.figure(figsize=(10,10))
            pyplot.show()
            fig1.savefig(plotname+"_trainingset_scatter.png", bbox_inches="tight")
        else:
            pyplot.savefig(plotname+"_trainingset_scatter.png")

    y_pred = model.predict(X_test)

    mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
    r2s = sklearn.metrics.r2_score(y_test, y_pred)

    rmse = math.sqrt(mse)
    if verbose:
        print("             Testset RMSE: %10.5f"%rmse, file=pout)
        print("                       R2: %10.5f"%r2s, file=pout)

    if verbose:
        pyplot.clf()
        pyplot.rcParams.update({'font.size': 15})
        pyplot.figure(figsize=(10,10))
        pyplot.title("Scatterplot for Test Set")
        pyplot.xlabel("log(Cases/Population)")
        pyplot.ylabel("Predicted log(Cases/Population)")
        pyplot.scatter(y_test, y_pred)

        if showplot:
            fig1 = pyplot.gcf()
            pyplot.figure(figsize=(10,10))
            pyplot.show()
            fig1.savefig(plotname+"_testset_scatter.png", bbox_inches="tight")
        else:
            pyplot.savefig(plotname+"_testset_scatter.png")

    results= permutation_importance(model, X_test, y_test, n_repeats=50, random_state=0, \
        scoring="neg_mean_squared_error")

    importance = results.importances_mean
    importanceerror = results.importances_std
    # summarize feature importance

    if verbose:
        print("",file=pout)
        print("Features importance from Permutation TetSet Score neg_mean_squared_error : ",file=pout)

    refvalue = 0.0
    for i,v in enumerate(importance):
        if features[i] == "randomfeature":
            refvalue = v

    totfi = 0.0
    featimport3 = {}
    for i,v in enumerate(importance):
        featimport3[features[i]] = v
        if verbose:
            if (v > 0.0 and v > refvalue):
                print('Feature: %30s, Score: %.5f +/- %.5f' % \
                    (featuresforplot[i],v,importanceerror[i]),file=pout)
        totfi += v

    for i,v in enumerate(importance):
        featimport3[features[i]] /= totfi

    if verbose:
        # plot feature importance
        pyplot.clf()
        pyplot.figure(figsize=(10,10))
        pyplot.title("Features importance from Permutation Test Set [Neg. MSE]")
        pyplot.barh(featuresforplot, importance, xerr=importanceerror, capsize=10)
        pyplot.xticks(rotation=45, ha="right")
        pyplot.gcf().subplots_adjust(bottom=0.30)
        if showplot:
            fig1 = pyplot.gcf()
            pyplot.figure(figsize=(10,10))
            pyplot.show()
            fig1.savefig(plotname+"_testset_feats_imp_frompermutation_neg_mean_squared_error.png", bbox_inches="tight")
        else:
            pyplot.savefig(plotname+"_testset_feats_imp_frompermutation_neg_mean_squared_error.png")

    results= permutation_importance(model, X_test, y_test, n_repeats=50, random_state=0, \
        scoring="r2")

    importance = results.importances_mean
    importanceerror = results.importances_std
    # summarize feature importance

    if verbose:
        print("",file=pout)
        print("Features importance from Permutation TestSet Score r2: ",file=pout)

    refvalue = 0.0
    for i,v in enumerate(importance):
        if features[i] == "randomfeature":
            refvalue = v

    totfi = 0.0
    featimport4 = {}
    for i,v in enumerate(importance):
        featimport4[features[i]] = v
        if verbose:
            if (v > 0.0 and v > refvalue):
                print('Feature: %30s, Score: %.5f +/- %.5f' % \
                    (featuresforplot[i],v,importanceerror[i]),file=pout)
        totfi += v

    for i,v in enumerate(importance):
        featimport4[features[i]] /= totfi

    if verbose:
        # plot feature importance
        pyplot.clf()
        pyplot.figure(figsize=(10,10))
        pyplot.title("Features importance from Permutation Test Set [R2]")
        pyplot.barh(featuresforplot, importance, xerr=importanceerror, capsize=10)
        pyplot.xticks(rotation=45, ha="right")
        pyplot.gcf().subplots_adjust(bottom=0.30)
        if showplot:
            fig1 = pyplot.gcf()
            pyplot.figure(figsize=(10,10))
            pyplot.show()
            fig1.savefig(plotname+"_testset_feats_imp_frompermutation_r2.png", bbox_inches="tight")
        else:
            pyplot.savefig(plotname+"_testset_feats_imp_frompermutation_testset_r2.png")


    results = permutation_importance(model, X_train, y_train, n_repeats=50, random_state=0, \
        scoring="neg_mean_squared_error")

    importance = results.importances_mean
    importanceerror = results.importances_std
    # summarize feature importance

    if verbose:
        print("",file=pout)
        print("Features importance from Permutation TrainingSet Score neg_mean_squared_error : ",file=pout)

    refvalue = 0.0
    for i,v in enumerate(importance):
        if features[i] == "randomfeature":
            refvalue = v

    totfi = 0.0
    featimport5 = {}
    for i,v in enumerate(importance):
        featimport5[features[i]] = v
        if verbose:
            if (v >0.0 and v > refvalue):
                print('Feature: %30s, Score: %.5f +/- %.5f' % \
                    (featuresforplot[i],v,importanceerror[i]),file=pout)
        totfi += v

    for i,v in enumerate(importance):
        featimport3[features[i]] /= totfi

    if verbose:
        # plot feature importance
        pyplot.clf()
        pyplot.figure(figsize=(10,10))
        pyplot.title("Features importance from Permutation Training Set [Neg. MSE]")
        pyplot.barh(featuresforplot, importance, xerr=importanceerror, capsize=10)
        pyplot.xticks(rotation=45, ha="right")
        pyplot.gcf().subplots_adjust(bottom=0.30)
        if showplot:
            fig1 = pyplot.gcf()
            pyplot.figure(figsize=(10,10))
            pyplot.show()
            fig1.savefig(plotname+"_trainingset_feats_imp_frompermutation_neg_mean_squared_error.png", bbox_inches="tight")
        else:
            pyplot.savefig(plotname+"_trainingset_feats_imp_frompermutation_neg_mean_squared_error.png")

    results= permutation_importance(model, X_train, y_train, n_repeats=50, random_state=0, \
        scoring="r2")

    importance = results.importances_mean
    importanceerror = results.importances_std
    # summarize feature importance

    if verbose:
        print("",file=pout)
        print("Features importance from Permutation TrainingSet Score r2: ",file=pout)

    refvalue = 0.0
    for i,v in enumerate(importance):
        if features[i] == "randomfeature":
            refvalue = v

    totfi = 0.0
    featimport6 = {}
    for i,v in enumerate(importance):
        featimport6[features[i]] = v
        if verbose:
            if (v > 0.0 and v > refvalue):
                print('Feature: %30s, Score: %.5f +/- %.5f' % \
                    (featuresforplot[i],v,importanceerror[i]),file=pout)
        totfi += v

    for i,v in enumerate(importance):
        featimport4[features[i]] /= totfi

    if verbose:
        # plot feature importance
        pyplot.clf()
        pyplot.figure(figsize=(10,10))
        pyplot.title("Features importance from Permutation Training Set [R2]")
        pyplot.barh(featuresforplot, importance, xerr=importanceerror, capsize=10)
        pyplot.xticks(rotation=45, ha="right")
        pyplot.gcf().subplots_adjust(bottom=0.30)
        if showplot:
            fig1 = pyplot.gcf()
            pyplot.figure(figsize=(10,10))
            pyplot.show()
            fig1.savefig(plotname+"_trainingset_feats_imp_frompermutation_r2.png", bbox_inches="tight")
        else:
            pyplot.savefig(plotname+"_trainingset_feats_imp_frompermutation_r2.png")

    return trainavgrmse, testavgrmse, fullsetrmse, featimport, featimport2,  \
        featimport3, featimport4, featimport5, featimport6

##################################################################################33

def rfregressors_custom_optimizer_split_testtr (Xin, yin, NSPLIT=10, \
    verbose=True, inboot=[True, False]):

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

                                print("%5d of %5d"%(counter, total))

                                diffstdperc_l = []
                                train_rmse_l = []
                                test_rmse_l = []
                                diffrmse_l = []
                                diffstdperc_l = []

                                for ns  in range(NSPLIT):
                                    X_train, X_test, y_train, y_test = train_test_split(
                                        Xin, yin, test_size=0.35)

                                    model = RandomForestRegressor(
                                        n_estimators=a,
                                        max_depth=b,
                                        min_samples_split=c,
                                        min_samples_leaf=d,
                                        random_state=e,
                                        bootstrap=f,
                                        max_features=g
                                    )

                                    model.fit(X_train, y_train)

                                    y_pred = model.predict(X_train)
                                    mse = sklearn.metrics.mean_squared_error(y_train, y_pred)
                                    train_rmse_l.append(math.sqrt(mse))
                                
                                    y_pred = model.predict(X_test)
                                    mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
                                    test_rmse_l.append(math.sqrt(mse))
 
                                    diffrmse_l.append(math.fabs(test_rmse_l[-1] -train_rmse_l[-1]))

                                    model1 = RandomForestRegressor(
                                        n_estimators=a,
                                        max_depth=b,
                                        min_samples_split=c,
                                        min_samples_leaf=d,
                                        random_state=e,
                                        bootstrap=f,
                                        max_features=g
                                    )

                                    model1.fit(Xin, yin)

                                    y_pred = model1.predict(Xin)

                                    diffstdperc_l.append(100*(math.fabs(np.std(y_pred) - np.std(yin))/np.std(yin)))

                                #print(counter , " of ", total ,"Train RMSE: ", train_rmse)
                                #print(counter , " of ", total ," Test RMSE: ", test_rmse)

                                diffstdperc = np.average(diffstdperc_l)
                                train_rmse = np.average(train_rmse_l)
                                test_rmse = np.average(test_rmse_l)
                                diffrmse = np.average(diffrmse_l)
                                diffstdperc = np.average(diffstdperc_l)

                                counter += 1

                                if diffstdperc < 80.0 and \
                                    train_rmse < best_train_rmse:

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

def rfregressors_multitestset (Xin, yin, features, plotname="rf_model", N = 50, 
    pout=sys.stdout, optimisedparams=None, visualmap=None, NFI=50, NJ=4):

    train_rmse = []
    train_r2 = []
    test_rmse = []
    test_r2 = []

    featuresforplot = []

    if visualmap == None:
        featuresforplot = features
    else:
        for f in features:
          vname = visualmap[f]
          featuresforplot.append(vname)

    test_featuresimportancenegmse = {}
    test_featuresimportancer2 = {}
    test_featuresimportancenegmse_first = {}
    test_featuresimportancer2_first = {}
    test_featuresimportancenegmse_second = {}
    test_featuresimportancer2_second = {}

    train_featuresimportancenegmse = {}
    train_featuresimportancer2 = {}
    train_featuresimportancenegmse_first = {}
    train_featuresimportancer2_first = {}
    train_featuresimportancenegmse_second = {}
    train_featuresimportancer2_second = {}

    for f in featuresforplot:
        train_featuresimportancenegmse[f] = []
        train_featuresimportancer2[f] = []

        test_featuresimportancenegmse_first[f] = 0
        test_featuresimportancer2_first[f] = 0
        test_featuresimportancenegmse_second[f] = 0
        test_featuresimportancer2_second[f] = 0

        test_featuresimportancenegmse[f] = []
        test_featuresimportancer2[f] = []

        train_featuresimportancenegmse_first[f] = 0
        train_featuresimportancer2_first[f] = 0
        train_featuresimportancenegmse_second[f] = 0
        train_featuresimportancer2_second[f] = 0

    for isplit in range(N):

        print("%5d of %5d"%(isplit, N))

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
        r2train = sklearn.metrics.r2_score(y_train, y_pred)
        train_r2.append(r2train)
        
        y_pred = model.predict(X_test)
        mse = sklearn.metrics.mean_squared_error(y_test, y_pred)
        rmse = math.sqrt(mse)
        test_rmse.append(rmse)
        r2test = sklearn.metrics.r2_score(y_test, y_pred)
        test_r2.append(r2test)

        results = permutation_importance(model, X_test, y_test, n_repeats=NFI, \
            scoring="neg_mean_squared_error", n_jobs=NJ)
        importance = results.importances_mean
        #importanceerror = results.importances_std
        max = float('-inf')
        max_f = ""
        second_last = float('-inf')
        second_last_f = ""
        for i,v in enumerate(importance):
            test_featuresimportancenegmse[featuresforplot[i]].append(v)
            if v > max:
                second_last = max
                max = v
                max_f = featuresforplot[i]
            elif v > second_last and v != max:
                second_last = v
                second_last_f = featuresforplot[i]

        test_featuresimportancenegmse_first[max_f] += 1
        if second_last_f != "":
            test_featuresimportancenegmse_second[second_last_f] += 1
        results = permutation_importance(model, X_test, y_test, n_repeats=NFI, \
            scoring="r2", n_jobs=NJ)
        importance = results.importances_mean
        #importanceerror = results.importances_std
        max = float('-inf')
        max_f = ""
        second_last = float('-inf')
        second_last_f = ""
        for i,v in enumerate(importance):
            test_featuresimportancer2[featuresforplot[i]].append(v)
            if v > max:
                second_last = max
                max = v
                max_f = featuresforplot[i]
            elif v > second_last and v != max:
                second_last = v
                second_last_f = featuresforplot[i]
        test_featuresimportancer2_first[max_f] += 1
        if second_last_f != "":
            test_featuresimportancer2_second[second_last_f] += 1

        results = permutation_importance(model, X_train, y_train, n_repeats=NFI, \
            scoring="neg_mean_squared_error", n_jobs=NJ)
        importance = results.importances_mean
        #importanceerror = results.importances_std
        max = float('-inf')
        max_f = ""
        second_last = float('-inf')
        second_last_f = ""
        for i,v in enumerate(importance):
            train_featuresimportancenegmse[featuresforplot[i]].append(v)
            if v > max:
                second_last = max
                max = v
                max_f = featuresforplot[i]
            elif v > second_last and v != max:
                second_last = v
                second_last_f = featuresforplot[i]
        train_featuresimportancenegmse_first[max_f] += 1
        if second_last_f != "":
            train_featuresimportancenegmse_second[second_last_f] += 1
        results = permutation_importance(model, X_test, y_test, n_repeats=NFI, \
            scoring="r2", n_jobs=NJ)
        importance = results.importances_mean
        #importanceerror = results.importances_std
        max = float('-inf')
        max_f = ""
        second_last = float('-inf')
        second_last_f = ""
        for i,v in enumerate(importance):
            train_featuresimportancer2[featuresforplot[i]].append(v)
            if v > max:
                second_last = max
                max = v
                max_f = featuresforplot[i]
            elif v > second_last and v != max:
                second_last = v
                second_last_f = featuresforplot[i]
        train_featuresimportancer2_first[max_f] += 1
        if second_last_f != "":
            train_featuresimportancer2_second[second_last_f] += 1

    trainavgrmse = (np.average(train_rmse), np.std(train_rmse))
    testavgrmse = (np.average(test_rmse), np.std(test_rmse)) 

    trainavgr2 = (np.average(train_r2), np.std(train_r2))
    testavgr2 = (np.average(test_r2), np.std(test_r2)) 

    print("Training set average RMSE: %8.5f +/- %8.5f "%(trainavgrmse[0], trainavgrmse[1]), 
            file=pout)
    print("    Test set average RMSE: %8.5f +/- %8.5f "%(testavgrmse[0], testavgrmse[1]),
            file=pout)

    print("  Training set average R2: %8.5f +/- %8.5f "%(trainavgr2[0], trainavgr2[1]), 
            file=pout)
    print("      Test set average R2: %8.5f +/- %8.5f "%(testavgr2[0], testavgr2[1]),
            file=pout)


    refval1 = np.average(train_featuresimportancenegmse["Random Feat."])
    refval2 = np.average(train_featuresimportancer2["Random Feat."])

    print("Taining:")
    for f in featuresforplot:
        val1average = np.average(train_featuresimportancenegmse[f])
        val2average = np.average(train_featuresimportancer2[f])
        if (val1average > 0.0 and val1average > refval1 and \
            val2average > 0.0 and val2average > refval2):
            print("%20s , %10.5f +/- %10.5f , %10.5f +/- %10.5f , %10.5f , %10.5f , %10.5f , %10.5f"%(f, \
                 val1average, 
                 np.std(train_featuresimportancenegmse[f]), 
                 val2average, 
                 np.std(train_featuresimportancer2[f]),
                 100.0*(train_featuresimportancenegmse_first[f]/N), 
                 100.0*(train_featuresimportancenegmse_second[f]/N), 
                 100.0*(train_featuresimportancer2_first[f]/N), 
                 100.0*(train_featuresimportancer2_second[f]/N)))

    refval1 = np.average(test_featuresimportancenegmse["Random Feat."])
    refval2 = np.average(test_featuresimportancer2["Random Feat."])

    print("Test:")
    for f in featuresforplot:
        val1average = np.average(test_featuresimportancenegmse[f])
        val2average = np.average(test_featuresimportancer2[f])
        if (val1average > 0.0 and val1average > refval1 and \
            val2average > 0.0 and val2average > refval2):
            print("%20s , %10.5f +/- %10.5f , %10.5f +/- %10.5f , %10.5f , %10.5f , %10.5f , %10.5f"%(f, \
                 val1average, 
                 np.std(test_featuresimportancenegmse[f]), 
                 val2average, 
                 np.std(test_featuresimportancer2[f]),  
                 100.0*(test_featuresimportancenegmse_first[f]/N), 
                 100.0*(test_featuresimportancenegmse_second[f]/N), 
                 100.0*(test_featuresimportancer2_first[f]/N), 
                 100.0*(test_featuresimportancer2_second[f]/N)))

    return 

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
