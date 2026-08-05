import pandas as pd
cad = pd.read_csv("notebooks/cad_cia_aberta.csv", sep=";", encoding="latin1")
cad.head()
cad.columns
cad_ativas = cad[cad["SIT"] == "ATIVO"]