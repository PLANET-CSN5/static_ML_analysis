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

from matplotlib import pyplot

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

def rfregressors (Xin, yin, features, plotname="RFmodel", N = 50, verbose=True):
    train_rmse = []
    test_rmse = []
 
    for isplit in range(N):
        X_train, X_test, y_train, y_test = train_test_split(
            Xin, yin, test_size=0.35)
        model = RandomForestRegressor()
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
      
    model = RandomForestRegressor()
 
    # fit the model
    model.fit(Xin, yin)
 
    y_pred = model.predict(Xin)
 
    mse = sklearn.metrics.mean_squared_error(yin, y_pred)
    rmse = math.sqrt(mse)
    if verbose:
        print("Fullset RMSE: ", rmse)
    
    fullsetrmse = rmse
 
    if verbose:
        pyplot.title("ScatterPlot predicted vs True")
        pyplot.scatter(yin, y_pred)
        #pyplot.show()
        pyplot.savefig(plotname+"_scatter.png")
 
    # get importance
    importance = model.feature_importances_
    if verbose:
        print("")
        print("Features importance from model: ")
        # summarize feature importance
        for i,v in enumerate(importance):
            print('Feature: %s, Score: %.5f' % (features[i],v))
 
        # plot feature importance
        pyplot.bar(features, importance)
        #pyplot.show()
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
    results = permutation_importance(model, Xin, yin, scoring='neg_mean_squared_error')
    # get importance
    importance = results.importances_mean
    # summarize feature importance

    if verbose:
        print("")
        print("Features importance from Permutation: ")

    totfi = 0.0
    featimport = {}
    for i,v in enumerate(importance):
        featimport[features[i]] = v
        if verbose:
            print('Feature: %s, Score: %.5f' % (features[i],v))
        totfi += v

    for i,v in enumerate(importance):
        featimport[features[i]] /= totfi

    if verbose:
        # plot feature importance
        pyplot.bar(features, importance)
        #pyplot.show()
        pyplot.savefig(plotname+"_feats_imp_frompermutation.png")

    return trainavgrmse, testavgrmse, fullsetrmse, featimport

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
