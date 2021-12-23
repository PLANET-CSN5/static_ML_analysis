import pandas as pd
import numpy as np 
import datetime

tc = pd.ExcelFile("/usr/local/share/public/TabelleCodici.xlsx")
data = pd.read_csv("/usr/local/share/public/ISS-COVID19.csv")

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
