import pandas as pd
import numpy as np 
import datetime

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


