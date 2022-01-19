import pandas as pd
import numpy as np 
import argparse
import math

import smlmodule
from matplotlib import pyplot

from itertools import combinations

__provmaps__ = {
    "bolzano_bozen": "bolzano",
    "bolzanobozen": "bolzano",
    "vibovalentia": "vibo_valentia",
    "laquila": "l_aquila",
    "laspezia": "la_spezia",
    "barlettaandriatrani": "bat",
    "ascolipiceno": "ascoli_piceno",
    "carboniaiglesias": "carbonia",
    "reggioemilia": "reggio_nell_emilia",
    "pesarourbino": "pesaro",
    "monzabrianza": "monza",
    "reggiocalabria": "reggio_di_calabria",
    "forlicesena": "forli",
    "massacarrara": "massa",
    "verbanocusioossola": "verbania",
    "verbano_cusio_ossola": "verbania",
    "massa_carrara": "massa",
    "monza_e_della_brianza": "monza",
    "pesaro_e_urbino": "pesaro",
    "forli__cesena": "forli",
    "barletta_andria_trani": "bat",
    "sud_sardegna": "carbonia"
}

############################################################################################

def filterprovname (inprov):
    low = inprov.lower()
    low = low.rstrip()
    low = low.lstrip()
    low = low.replace(" ", "_")
    low = low.replace("'", "_")
    low = low.replace("-", "_")

    return low

############################################################################################

def normalize_provname (indata, provcolumn, verbose):

    dict_data = {}  
    for c in indata.columns:
        if verbose:
            print("  ", c)
        if c != provcolumn:
            dict_data[c] = []
    dict_data["prov"] = []

    for i, row in indata.iterrows():
        for c in indata.columns:    
            if c != provcolumn:
                dict_data[c].append(row[c])
            else:
                low = filterprovname(row[c])
                if low in __provmaps__:
                    low = __provmaps__[low]

                dict_data["prov"].append(low)

    #for v in dict_data:
    #    print(v, " ", len(dict_data[v]))

    data = pd.DataFrame.from_dict(dict_data)

    return data

############################################################################################

if __name__ == "__main__":

    paperpath = "particulate.csv"
    labelspath = "2020_2_24_to_2020_3_20.csv"
    deprividxpath = "ID11_prov21.xlsx"
    copernicopath = "name_region_province_statistics_2020.csv"

    pollutantsnames = "avg_wco_period1_2020,"+\
        "avg_wnh3_period1_2020,"+\
        "avg_wnmvoc_period1_2020,"+\
        "avg_wno2_period1_2020,"+\
        "avg_wno_period1_2020,"+\
        "avg_wo3_period1_2020,"+\
        "avg_wpans_period1_2020,"+\
        "avg_wpm10_period1_2020,"+\
        "avg_wpm2p5_period1_2020,"+\
        "avg_wso2_period1_2020"

    parser = argparse.ArgumentParser()

    parser.add_argument("--labels-file", help="Specify Labels CSV file default: " +  labelspath , \
        type=str, required=False, default=labelspath, dest="labelspath")
    parser.add_argument("--paper-data-file", help="Specify Setti Paper data CSV file default: " + paperpath , \
        type=str, required=False, default=paperpath, dest="paperpath")
    parser.add_argument("--depriv-index-file", help="Specify Depriv Index Excel file default: " + deprividxpath , \
        type=str, required=False, default=deprividxpath, dest="deprivpath")
    parser.add_argument("--copernico-data", help="Specify Copernico data file: " + copernicopath , \
        type=str, required=False, default=copernicopath, dest="copernicopath")
    parser.add_argument("-v", "--verbose", help="Increase output verbosity", \
        default=False, action="store_true")
    parser.add_argument("-c", "--checkdetails", help="Check a single models all details dump graphs", \
        default=False, action="store_true")
    parser.add_argument("-p", "--pollutantsnames", help="List of pollutants to be used comma separated default: " + \
        pollutantsnames , default=pollutantsnames, type=str)
    args = parser.parse_args()

    in_issdata = pd.read_csv(args.labelspath)
    in_datapaper = pd.read_csv(args.paperpath, sep=";")
    in_deprividx =  pd.ExcelFile(args.deprivpath).parse("Foglio1")
    in_copernico = pd.read_csv(args.copernicopath)

    print("ISS data ") 
    issdata = normalize_provname(in_issdata, "prov", args.verbose)

    print("Copernico data ") 
    copernico = normalize_provname(in_copernico, "nome_ita", args.verbose)

    print("Paper data ")
    datapaper = normalize_provname(in_datapaper, "Province", args.verbose)
    
    dict_deprividx = {}
    print("DrepivIdx name ")
    for c in in_deprividx.columns:
        if args.verbose:
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

    #get unique provice list
    provincelist = list(set(list(issdata["prov"].values)) & \
        set(list(datapaper["prov"].values)) & \
        set(list(deprividx["prov"].values)) & \
        set(list(copernico["prov"].values)))

    print("Province list: ")
    for i, p in enumerate(provincelist):
        print("  ", i+1, " ", p)

    #issdata_dict = {}
    # check on y values 
    for i, row in issdata.iterrows():

        prov = row["prov"]
        
        ycasi = row["dataprelievo"]
        ysintomi = row["sintomatici_dataprelievo"]
        ydeceduti = row["deceduti_dataprelievo"]
        yricoverati = row["ricoverati_dataprelievo"]
        yterapiaintensiva = row["terapiaintensiva_dataprelievo"]
        
        #issdata_dict[prov] = { \
        #    "casi" : ycasi, \
        #    "casi_deceduti" : ydeceduti, \
        #    "casi_ricoverati" : yricoverati, \
        #    "casi_con_sintomi" : ysintomi, \
        #    "casi_terapiaintensiva" : yterapiaintensiva}
        
        if  ycasi < ydeceduti or \
            ycasi < ysintomi or \
            ycasi < yricoverati or \
            ycasi < yterapiaintensiva:
            print("%25s %5d %5d %5d %5d "%(prov, ycasi, \
                                           ydeceduti, \
                                           ysintomi, \
                                           yricoverati, \
                                           yterapiaintensiva))
            exit(1)
    for label in ["dataprelievo", \
              "sintomatici_dataprelievo", \
              "deceduti_dataprelievo", \
              "ricoverati_dataprelievo", \
              "terapiaintensiva_dataprelievo"]:

        print("Label: ", label)
        
        features_dict = {}
        ylogpropcasi = []
        ypropcasi = []
        
        counter = 0
        for i, prov in enumerate(provincelist):
            y = issdata[issdata["prov"] == prov][label].values[0]
            if y > 0.0:
                popolazione  = datapaper[datapaper["prov"] == prov]["Population"].values[0]
                ylogpropcasi.append(math.log(y/popolazione))
                ypropcasi.append(y/popolazione)
                counter += 1
                #print(i+1, " ", prov, " ", y, " ", popolazione)
        print("  ", counter, " active province")
        
        # non pollutants features
        for fn in ("population", "density", "commutersdensity", "depriv", "lat"):
            features_dict[fn] = np.zeros(counter, dtype="float64")
            
        i = 0
        for prov in provincelist:
            y = issdata[issdata["prov"] == prov][label].values[0]
            if y > 0.0:
                popolazione  = datapaper[datapaper["prov"] == prov]["Population"].values[0]
                features_dict["population"][i] = popolazione
                features_dict["density"][i] = \
                    datapaper[datapaper["prov"] == prov]["Density"].values[0]    
                features_dict["commutersdensity"][i] = \
                    datapaper[datapaper["prov"] == prov]["CommutersDensity"].values[0]       
                features_dict["lat"][i] = \
                    datapaper[datapaper["prov"] == prov]["Lat"].values[0]       
                features_dict["depriv"][i] = \
                    deprividx[deprividx["prov"] == prov]["ID_2011"].values[0]
                
                i = i + 1
                
        # polluttats features
        for fn in pollutantsnames.split(","):
            features_dict[fn] = np.zeros(counter, dtype="float64")
            
        i = 0
        for prov in provincelist:
            y = issdata[issdata["prov"] == prov][label].values[0]
            if y > 0.0:
                selected = copernico[copernico["prov"] == prov]
 
                for fn in pollutantsnames.split(","):
                    val = selected[fn].values[0]
                    features_dict[fn][i] = val 
                
                i = i + 1

        fullfeatset = []
        for fn in pollutantsnames.split(","):
            fullfeatset.append(fn)
        fullfeatset.extend(["density", "commutersdensity", "depriv", "lat"])
        y = ylogpropcasi
        print("")
        print("Method , Avg. Train RMSE , Std. , Avg. Test RMSE , Std. , Full RMSE , ", end ="")
        for i, f in enumerate(fullfeatset):
            print (f + " , ", end="")
        print(", Top ranked Features")
        
        features = fullfeatset
        listostack = [features_dict[v] for v in features]
        X = np.column_stack (listostack) 
        
        rf = smlmodule.rfregressors (X, y, features, verbose=False)
        #kn = knregressors (X, y, features)
        smlmodule.printcsvRF (fullfeatset, features, rf)

""" 
    prinvincewithzero = set() 
    province = datapaper["prov"].values
 
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
        popolazione  = datapaper[datapaper["prov"] == prov]["Population"].values[0]
        
        features_dict["pm10"][i] = \
          datapaper[datapaper["prov"] == prov]["9_29_feb_0.0_mean_pm10_ug/m3_2020"].values[0]
        features_dict["pm25"][i] = \
          datapaper[datapaper["prov"] == prov]["9_29_feb_0.0_mean_pm2p5_ug/m3_2020"].values[0]
        features_dict["pm10ts"][i] = \
          datapaper[datapaper["prov"] == prov]["9_29_feb_0.0_std_ts_pm10_n_2020"].values[0]   
        features_dict["pm25ts"][i] = \
          datapaper[datapaper["prov"] == prov]["9_29_feb_0.0_std_ts_pm2p5_n_2020"].values[0] 
 
        features_dict["popolation"][i] = popolazione
        features_dict["density"][i] = \
          datapaper[datapaper["prov"] == prov]["Density"].values[0]    
        features_dict["commutersdensity"][i] = \
          datapaper[datapaper["prov"] == prov]["CommutersDensity"].values[0]       
        features_dict["lat"][i] = \
          datapaper[datapaper["prov"] == prov]["Lat"].values[0]       
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
""" 