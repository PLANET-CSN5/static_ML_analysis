import pandas as pd
import numpy as np 
import argparse
import math
import sys

from scipy.stats import pearsonr
import matplotlib.pyplot as plt 

import smlmodule

from itertools import combinations

LIMIT = 0.9

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

    """
    period1 = ['2020-02-09', '2020-02-28'] # YEAR-MONTH-DAY --->>> CASI COVID ['2020-02-24', '2020-03-13']
    period2 = ['2020-02-09', '2020-03-06] # YEAR-MONTH-DAY --->>> CASI COVID ['2020-02-09', '2020-03-21']
    period3 = ['2020-08-29', '2020-09-01'] # YEAR-MONTH-DAY --->>> CASI COVID ['2020-09-12', '2020-10-15']
    period4 = ['2020-08-29', '2020-10-30'] # YEAR-MONTH-DAY --->>> CASI COVID ['2020-09-12', '2020-11-14']
    period5 = ['2020-05-15', '2020-08-15'] # YEAR-MONTH-DAY --->>> CASI COVID ['2020-06-01', '2020-09-01']
    """

    paperpath = "particulate.csv"
    labelspath = "2020_2_24_period1_to_2020_3_13.csv"
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
        "avg_wso2_period1_2020," +\
        "sum_wnh3_ex_q75_period1_2020," +\
        "sum_wnmvoc_ex_q75_period1_2020," +\
        "sum_wno2_ex_q75_period1_2020," +\
        "sum_wno_ex_q75_period1_2020," +\
        "sum_wpans_ex_q75_period1_2020," +\
        "sum_wpm10_ex_q75_period1_2020," +\
        "sum_wpm2p5_ex_q75_period1_2020," +\
        "sum_wso2_ex_q75_period1_2020"
#       "sum_wo3_ex_q75_period1_2020," +\
#       "sum_wco_ex_q75_period1_2020," +\ all zeros

    allpossiblelabels = "dataprelievo," + \
        "sintomatici_dataprelievo,"+ \
        "deceduti_dataprelievo,"+ \
        "ricoverati_dataprelievo,"+ \
        "terapiaintensiva_dataprelievo"

    featurestobeused = "density," + \
        "commutersdensity," + \
        "lat," + \
        "depriv," + \
        "avg_wpm10_period1_2020,"+\
        "avg_wpm2p5_period1_2020,"+\
        "avg_wco_period1_2020,"+\
        "avg_wnh3_period1_2020,"+\
        "avg_wnmvoc_period1_2020,"+\
        "avg_wno2_period1_2020,"+\
        "avg_wno_period1_2020,"+\
        "avg_wo3_period1_2020,"+\
        "avg_wpans_period1_2020,"+\
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
    parser.add_argument("-l", "--alllabels", help="List of all labels to be used comma separated default: " + \
        pollutantsnames , default=allpossiblelabels, type=str)
    parser.add_argument("--featstouse", help="List of all features to use will remove correlated ordered by priority comma separated default: " + \
        pollutantsnames , default=featurestobeused, type=str)
 
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


    for label in args.alllabels.split(","):

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

        # nomalize values
        for fn in features_dict:
            #print(fn)
            abs_max = np.amax(np.abs(features_dict[fn]))
            if abs_max == 0.0:
                print(fn, " ", features_dict[fn])
            features_dict[fn] = features_dict[fn] * (1.0 / abs_max)
        
        highcorrelated = {}
        for i1, v1 in enumerate(features_dict):
            highcorrelated[v1] = []
            for i2, v2 in enumerate(features_dict):
                #if v1 != v2 and i2 > i1:
                if v1 != v2:
                    corr, _ = pearsonr(features_dict[v1], features_dict[v2])
                    if math.fabs(corr) > LIMIT:
                        highcorrelated[v1].append(v2)
                        #print(v1, v2, corr)

            #if len(highcorrelated[v1]) > 0:
            #    print(v1)
            #    for fntr in highcorrelated[v1]:
            #        print("   ", fntr)

        features = []
        for fn in featurestobeused.split(","):
            canadd = True
            for fnin in features:
                if fn in highcorrelated[fnin]:
                    canadd = False
                    break

            if canadd:
                print("Using: ", fn)
                features.append(fn)
            #else:
            #    print(fn, " correlated removing")

        listostack = [features_dict[v] for v in features]
        X = np.column_stack (listostack)
 
        Y = ylogpropcasi
        #print(y.shape)
        plt.figure(figsize=(5,5))
        smlmodule.rfregressors (X, Y , features, plotname="RFmodel_"+label, N=50)
        #smlmodule.knregressors (X, Y , features, N=50)
        

