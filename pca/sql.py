import sqlite3
import re

def write_data(filename):
    conn = sqlite3.connect('world-development-indicators/database.sqlite')
    c = conn.cursor()
    file_to_write = open(filename, 'w')
    rx = re.compile("\(|\)|'")
    for row in c.execute(
                        # SQL statement 
                        """
                            SELECT   CountryCode, IndicatorCode, Value 
                            FROM     Indicators 
                            WHERE    Year=2013
                         """ ):
        file_to_write.write(re.sub(rx, '', str(row)) + "\n")
    file_to_write.close()

#write_data('recent_compact_2013.csv')

def gni_graph(filename):
    conn = sqlite3.connect('../database.sqlite')
    c = conn.cursor()
    file_to_write = open(filename, 'w')
    rx = re.compile("\(|\)|'")
    for row in c.execute(
                        # SQL statement 
                        """
                            SELECT   CountryCode, Year, Value 
                            FROM     Indicators 
                            WHERE    IndicatorCode="NY.GNP.PCAP.KD";
                         """ ):
        file_to_write.write(re.sub(rx, '', str(row)) + "\n")
    file_to_write.close()

gni_graph("gni_test_file.csv")