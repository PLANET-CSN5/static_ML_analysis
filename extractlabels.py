import pandas as pd
import numpy as np 
import datetime
import argparse

import smlmodule

if __name__ == "__main__":

    tabellecodicipath = "/usr/local/share/public/TabelleCodici.xlsx"
    issdatacsvpath = "/usr/local/share/public/ISS-COVID19.csv"

    parser = argparse.ArgumentParser()

    parser.add_argument("--tabelle-codici-file", help="Specify TabelleCodici Excel file default: " + tabellecodicipath , \
        type=str, required=False, default=tabellecodicipath, dest="tcpath")
    parser.add_argument("--ISS-data-file", help="Specify ISS data CSV file default: " + issdatacsvpath , \
        type=str, required=False, default=issdatacsvpath, dest="isspath")
    
    args = parser.parse_args()

    tc = pd.ExcelFile(args.tcpath)
    data = pd.read_csv(args.isspath)
    
    print("Columns of ISS data: ")
    for c in data.columns:
        print ("  ", c)
    print("Sheets name of Tablla Codici: ")
    for c in tc.sheet_names:
        print ("  ", c)
    
    idtoprov = {}
    province = tc.parse("Codice Provincia")
    for val in province[["Codice Provincia","Nome Provincia"]].values:
        if type(val[1]) != float:
            idtoprov[int(val[0])] = val[1]
            #print(int(val[0]), val[1])
    
    #for id in idtoprov:
    #    print(id , " ==> ", idtoprov[id])
    #    newdata = extract_given_prov (data, id)
    #    newdata.to_csv(idtoprov[id]+".csv")
    
    # check casoimportato
    print("Possible values of CASOIMPORTATO ", set(data["CASOIMPORTATO"].values))
    unique, counts = np.unique(data["CASOIMPORTATO"].values, return_counts=True)
    print("   Counters ", dict(zip(unique, counts)))
    
    # rimuivo i casi importati 
    data = data[data.CASOIMPORTATO == "N"]
    
    print("Check extrated data CASOIMPORTATO values: ", set(data["CASOIMPORTATO"].values))
    unique, counts = np.unique(data["CASOIMPORTATO"].values, return_counts=True)
    print("Casi Non importati Totali", dict(zip(unique, counts)))
    
    # estraggo solo deceduti e solo terapieintensive solo ricoverati e solo sintomatici 
    deceduti = data[data.DECEDUTO == "Y"]
    terapiaintensiva = data[data.TERAPIAINTENSIVA == "Y"]
    ricoverati = data[data.RICOVERO == "Y"]
    sintomatici = data[data.SINTOMATICO == "Y"]
    
    print("Deceduti          :", len(deceduti))
    print("Terapia Intensiva :", len(terapiaintensiva))
    print("Ricoverati        :", len(ricoverati))
    print("Sintomatici       :", len(sintomatici))
    
    fpout = open("2402_to_1303.csv", "w")
    print("id,prov,dataprelievo,deceduti_dataprelievo," + \
       "ricoverati_dataprelievo,sintomatici_dataprelievo,terapiaintensiva_dataprelievo")
    fpout.write("id,prov,dataprelievo,deceduti_dataprelievo," + \
       "ricoverati_dataprelievo,sintomatici_dataprelievo,terapiaintensiva_dataprelievo\n")

    for id in idtoprov:
        newdata = smlmodule.extract_given_prov (data, id)

        sdate = datetime.date(2020, 2, 24)   # start date
        edate = datetime.date(2020, 3, 13)   # end date
       
        dataprelievo, datasintomi, datadiagnosi = smlmodule.extraxtnumber (data, id, sdate, edate)
        deceduti_dataprelievo, deceduti_datasintomi, deceduti_datadiagnosi = \
            smlmodule.extraxtnumber (deceduti, id, sdate, edate)
        terapiaintensiva_dataprelievo, terapiaintensiva_datasintomi, terapiaintensiva_datadiagnosi = \
            smlmodule.extraxtnumber (terapiaintensiva, id, sdate, edate)
        ricoverati_dataprelievo, ricoverati_datasintomi, ricoverati_datadiagnosi = \
            smlmodule.extraxtnumber (ricoverati, id, sdate, edate)
        sintomatici_dataprelievo, sintomatici_datasintomi,sintomatici_datadiagnosi = \
            smlmodule.extraxtnumber (sintomatici, id, sdate, edate)
       
        print(id,",",idtoprov[id],",",dataprelievo, \
              ",",deceduti_dataprelievo,\
              ",",ricoverati_dataprelievo,\
              ",",sintomatici_dataprelievo,\
              ",",terapiaintensiva_dataprelievo)
        
        fpout.write(str(id)+"," + \
                    str(idtoprov[id]) + "," + \
                    str(dataprelievo) + "," + \
                    str(deceduti_dataprelievo) + "," + \
                    str(ricoverati_dataprelievo) + "," + \
                    str(sintomatici_dataprelievo) + "," + \
                    str(terapiaintensiva_datadiagnosi) + "\n")
                    
    fpout.close()
