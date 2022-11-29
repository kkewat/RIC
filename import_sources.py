import sqlite3 as sq
import pandas as pd

Base='D:/practical-data-science-master/VKHCG'
sDatabaseName = Base +'/01-Vermeulen/04-Transform/SQLite/Vermeulen.db'
con = sq.connect(sDatabaseName)

sFileName = Base+'/01-Vermeulen/01-Retrieve/01-EDS/02-Python/Retrieve_IP_DATA.csv'
print('Loading :',sFileName)

IP_DATA_ALL_FIX = pd.read_csv(sFileName,header=0,low_memory=False)
IP_DATA_ALL_FIX.index.names = ["RowIDCSV"]
sTable='IP_DATA_ALL'

print('STORING :',sDatabaseName,' Table :',sTable)
IP_DATA_ALL_FIX.to_sql(sTable,con,if_exists="replace")
print('Loading :',sDatabaseName,'Table',sTable)
TestData = pd.read_sql_query("select * from IP_DATA_ALL;",con)
print('#'*30)
print('##Data values')
print('#'*30)
print('Test Data')
print('#'*30)
print('##Data Profile')
print('#'*30)
print('Rows :',TestData.shape[0])
print('Columns :',TestData.shape[1])
print('#'*30)
print('#'*15,'Done','#'*15)
