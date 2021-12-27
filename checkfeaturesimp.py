import pandas as pd
import numpy as np 
import argparse
import math

import smlmodule

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
    