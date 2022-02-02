import pandas as pd 

##############################################################################

def extractcity (data, cityname):
    
    citta = data[data["Territorio"] == cityname]
    cittatotal = citta[citta["ETA1"] == "TOTAL"]
    mfcittatotal = cittatotal[cittatotal["Sesso"] == "totale"]
    final = mfcittatotal[mfcittatotal["Stato civile"] == "totale"]
    
    results = final[final.ITTER107.str.contains("IT")]
    popolazione = results["Value"].values[0]
    
    return citta, popolazione

##############################################################################

def extraperage (citta, startage, endage=None):

    sum = 0.0

    if endage == None:
        val = "Y_GE100"
        citaETA = citta[citta["ETA1"] == val]
        MFcitaETA = citaETA[citaETA["Sesso"] == "totale"]
        final = MFcitaETA[MFcitaETA["Stato civile"] == "totale"]
        results = final[final.ITTER107.str.contains("IT")]
        #print(val, results["Value"].values[0])
        sum += results["Value"].values[0]
        endage = 99

    for i in range (startage,endage+1):
        val = "Y"+str(i)
        citaETA = citta[citta["ETA1"] == val]
        MFcitaETA = citaETA[citaETA["Sesso"] == "totale"]
        final = MFcitaETA[MFcitaETA["Stato civile"] == "totale"]
        results = final[final.ITTER107.str.contains("IT")]
        #print(val, results["Value"].values[0])
        sum += results["Value"].values[0]
    
    return sum 
    
##############################################################################

def extractfulldata (data, cityname):

    citta, popolazione = extractcity (data, cityname)
    
    totcheck = 0.0

    sum = extraperage(citta, 0, 20)
    #print(100.0 * (sum/popolazione))
    totcheck += sum

    bambini = sum

    sum = extraperage(citta, 21, 64)
    #print(100.0 * (sum/popolazione))
    totcheck += sum

    sum = extraperage(citta, 65)
    #print(100.0 * (sum/popolazione))
    totcheck += sum

    anziani = sum

    #print("Tot check: ", totcheck)
    if totcheck != popolazione:
       print(cityname, " WARNING ", cityname, totcheck, popolazione) 

    #print(anziani/bambini)

    return  anziani/bambini, popolazione

##############################################################################

if __name__ == "__main__":
    data = pd.read_csv("popolazione2020_onlyIT.csv", sep="|")

    listacitta = list(set(data["Territorio"].values))
    #print(listacitta)

    finaresults = {}
    finaresults["Provincia"] = []
    finaresults["Ratio0200ver65"] = []
    finaresults["Population2020"] = []
    for i, cityname in enumerate(listacitta):
        feat , pop = extractfulldata(data, cityname)
        finaresults["Provincia"].append(cityname)
        finaresults["Ratio0200ver65"].append(feat)
        finaresults["Population2020"].append(pop)
        #print(cityname, ",", extractfulldata(data, cityname))
        print(i+1 , " of ", len(listacitta))

    df = pd.DataFrame(finaresults)

    df.to_csv("provinceages.csv")