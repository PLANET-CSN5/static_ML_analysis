import pandas as pd
import numpy as np 
import argparse
import math

import smlmodule
from matplotlib import pyplot

from itertools import combinations

if __name__ == "__main__":

    labelspath = "2402_to_1303.csv"
    paperpath = "/usr/local/share/public/new_particulate_extended.csv"
    deprividxpath = "ID11_prov21.xlsx"

    parser = argparse.ArgumentParser()

    parser.add_argument("--labels-file", help="Specify Labels CSV file default: " +  labelspath , \
        type=str, required=False, default=labelspath, dest="labelsath")
    parser.add_argument("--paper-data-file", help="Specify Setti Paper data CSV file default: " + paperpath , \
        type=str, required=False, default=paperpath, dest="paperpath")
    parser.add_argument("--depriv-index-file", help="Specify Depriv Index Excel file default: " + deprividxpath , \
        type=str, required=False, default=deprividxpath, dest="deprivpath")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", \
        default=False, action="store_true")
    parser.add_argument("-c", "--checkdetails", help="Check a single models all detals dump graphs", \
        default=False, action="store_true")
    args = parser.parse_args()

    in_issdata = pd.read_csv(args.labelsath)
    datapaper = pd.read_csv(args.paperpath)
    in_deprividx =  pd.ExcelFile(args.deprivpath).parse("Foglio1")

    dict_issdata = {}  
    print("ISS data: ") 
    for c in in_issdata.columns:
        print("  ", c)
        dict_issdata[c] = []
        
    for i, row in in_issdata.iterrows():
        for c in in_issdata.columns:    
            if c != "prov":
                dict_issdata[c].append(row[c])
            else:
                low = row[c].lower()
                low = low.rstrip()
                low = low.lstrip()
                low = low.replace(" ", "_")
                low = low.replace("'", "_")
                low = low.replace("-", "_")
                
                dict_issdata[c].append(low)
    
    issdata = pd.DataFrame.from_dict(dict_issdata)
    
    #print(issdata[issdata["id"] == 1]["prov"].values[0])
    
    dict_deprividx = {}
    print("DrepivIdx name: ")
    for c in in_deprividx.columns:
        print("   ", c)   
        dict_deprividx[c] = []
    dict_deprividx["prov"] = []
    
    for i, row in in_deprividx.iterrows():
        id = row["prov21"]
        prov = issdata[issdata["id"] == id]["prov"].values[0]
        #print(id, prov)
        dict_deprividx["prov"].append(prov)
        for c in in_deprividx.columns:
            dict_deprividx[c].append(row[c])
    
    deprividx = pd.DataFrame.from_dict(dict_deprividx)
    
    if args.verbose:
        print("New Particolate: ")
        for c in datapaper.columns:
            print("   ", c)

    issdata_dict = {}
    
    for i, row in issdata.iterrows():

        prov = row["prov"]
        
        ycasi = row["dataprelievo"]
        ysintomi = row["sintomatici_dataprelievo"]
        ydeceduti = row["deceduti_dataprelievo"]
        yricoverati = row["ricoverati_dataprelievo"]
        yterapiaintensiva = row["terapiaintensiva_dataprelievo"]
        
        issdata_dict[prov] = { \
            "casi" : ycasi, \
            "casi_deceduti" : ydeceduti, \
            "casi_ricoverati" : yricoverati, \
            "casi_con_sintomi" : ysintomi, \
            "casi_terapiaintensiva" : yterapiaintensiva}
        
        if  ycasi < ydeceduti or \
            ycasi < ysintomi or \
            ycasi < yricoverati or \
            ycasi < yterapiaintensiva:
            print("%25s %5d %5d %5d %5d "%(low, ycasi, \
                                           ydeceduti, \
                                           ysintomi, \
                                           yricoverati, \
                                           yterapiaintensiva))


    prinvincewithzero = set() 
    province = datapaper["Province"].values

    features_dict = {}
    
    for fn in ("pm10", "pm25", "pm10ts", "pm25ts", "popolation", "density",\
              "commutersdensity", "depriv", "lat"):
        features_dict[fn] = np.zeros(len(province), dtype="float64")
    
    labelnames = ("casi", "casi_deceduti", \
                  "casi_con_sintomi", "casi_ricoverati", \
                  "casi_terapiaintensiva" )
    
    ylogpropcasi = {}
    ypropcasi = {}
    for ln in labelnames:
        ylogpropcasi[ln] = np.zeros(len(province))
        ypropcasi[ln] = np.zeros(len(province))
                                    
    for i, prov in enumerate(province):
        popolazione  = datapaper[datapaper["Province"] == prov]["Population"].values[0]
        
        features_dict["pm10"][i] = \
          datapaper[datapaper["Province"] == prov]["9_29_feb_0.0_mean_pm10_ug/m3_2020"].values[0]
        features_dict["pm25"][i] = \
          datapaper[datapaper["Province"] == prov]["9_29_feb_0.0_mean_pm2p5_ug/m3_2020"].values[0]
        features_dict["pm10ts"][i] = \
          datapaper[datapaper["Province"] == prov]["9_29_feb_0.0_std_ts_pm10_n_2020"].values[0]   
        features_dict["pm25ts"][i] = \
          datapaper[datapaper["Province"] == prov]["9_29_feb_0.0_std_ts_pm2p5_n_2020"].values[0] 
        features_dict["popolation"][i] = popolazione
        features_dict["density"][i] = \
          datapaper[datapaper["Province"] == prov]["Density"].values[0]    
        features_dict["commutersdensity"][i] = \
          datapaper[datapaper["Province"] == prov]["CommutersDensity"].values[0]       
        features_dict["lat"][i] = \
          datapaper[datapaper["Province"] == prov]["Lat"].values[0]       
        features_dict["depriv"][i] = \
          deprividx[deprividx["prov"] == prov]["ID_2011"].values[0]

        for ln in labelnames:
            ypropcasi[ln][i] = issdata_dict[prov][ln]/popolazione
            if (issdata_dict[prov][ln] == 0.0):
                #Note: Maybe to be removed 
                print("Zero %25s %25s "%(ln,prov))
                prinvincewithzero.add(prov)
            else:
                ylogpropcasi[ln][i] = math.log(ypropcasi[ln][i])
                    

    #print(y.shape)
    if args.checkdetails:
        features = ("pm10", "density", "commutersdensity", "depriv", "lat")
        listostack = [features_dict[v] for v in features]
        X = np.column_stack (listostack)

        #X = np.column_stack ((features_dict["pm10"], \
        #                  features_dict["pm25"], \
        #                  features_dict["density"], \
        #                  features_dict["commutersdensity"], \
        #                  features_dict["depriv"], \
        #                  features_dict["Lat"]))

        Y = ylogpropcasi["casi"]
        #print(y.shape)
        pyplot.figure(figsize=(5,5))
        smlmodule.rfregressors (X, Y , features, N=50)
        smlmodule.knregressors (X, Y , features, N=50)

    fullfeatset = ("pm10", "pm25", "density", "commutersdensity", "depriv", "lat")
    y = ylogpropcasi["casi"]
    print("")
    print("Method , Avg. Train RMSE , Std. , Avg. Test RMSE , Std. , Full RMSE , ", end ="")
    for i, f in enumerate(fullfeatset):
        print (f + " , ", end="")
    print(", Top ranked Features")

    features = ("pm10", "pm25", "density", "commutersdensity", "depriv", "lat")
    listostack = [features_dict[v] for v in features]
    X = np.column_stack (listostack) 
    
    rf = smlmodule.rfregressors (X, y, features, verbose=False)
    #kn = knregressors (X, y, features)
    smlmodule.printcsvRF (fullfeatset, features, rf)

    for fn in fullfeatset:
        features = []
        for v in fullfeatset:
            if v != fn:
                features.append(v)

        listostack = [features_dict[v] for v in features]
        X = np.column_stack (listostack)
        rf = smlmodule.rfregressors (X, y, features, verbose=False)
        smlmodule.printcsvRF (fullfeatset, features, rf)

    pairs = list(combinations(fullfeatset, 2))
    for p in pairs:
        features = []
        for v in fullfeatset:
            if v not in p:
                features.append(v)

        listostack = [features_dict[v] for v in features]
        X = np.column_stack (listostack)
        rf = smlmodule.rfregressors (X, y, features, verbose=False)
        smlmodule.printcsvRF (fullfeatset, features, rf)

    tris = list(combinations(fullfeatset, 3))
    for p in tris:
        features = []
        for v in fullfeatset:
            if v not in p:
                features.append(v)

        listostack = [features_dict[v] for v in features]
        X = np.column_stack (listostack)
        rf = smlmodule.rfregressors (X, y, features, verbose=False)
        smlmodule.printcsvRF (fullfeatset, features, rf)
