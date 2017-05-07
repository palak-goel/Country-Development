import sqlite3
import re
import os

#upload database.sqlite into this folder!!!

def write_data(filename, year="2013"):
    conn = sqlite3.connect('database.sqlite')
    c = conn.cursor()
    file_to_write = open(filename, 'w')
    rx = re.compile("\(|\)|'")
    statement = """
                            SELECT   CountryCode, IndicatorCode, Value 
                            FROM     Indicators 
                            WHERE    IndicatorCode=? AND Year="""+year
    for row in c.execute(statement, ['SP.DYN.LE00.IN']):
    # for row in c.execute(
    #                     # SQL statement 
    #                     """
    #                         SELECT   CountryCode, IndicatorCode, Value 
    #                         FROM     Indicators 
    #                         WHERE    IndicatorCode=[SP.DYN.LE00.IN] AND Year="""+year):
    
        file_to_write.write(re.sub(rx, '', str(row)) + "\n")
    file_to_write.close()

def write_multiple_data_files(base_filename, start_year=1960, end_year=2013):
    for year in range(start_year, end_year+1):
        filename = base_filename + "_" + str(year) + ".csv"
        write_data(filename, str(year))

write_multiple_data_files(os.path.abspath("life_expect/compact"))

# def gni_graph(filename):
#     conn = sqlite3.connect('../database.sqlite')
#     c = conn.cursor()
#     file_to_write = open(filename, 'w')
#     rx = re.compile("\(|\)|'")
#     for row in c.execute(
#                         # SQL statement 
#                         """
#                             SELECT   CountryCode, Year, Value 
#                             FROM     Indicators 
#                             WHERE    IndicatorCode="NY.GNP.PCAP.KD";
#                          """ ):
#         file_to_write.write(re.sub(rx, '', str(row)) + "\n")
#     file_to_write.close()

# gni_graph("gni_test_file.csv")