#!/usr/bin/env bash
: '
Process SQL data externally via Bash/Python

USAGE:  ./sqlextern.sh

OUTPUT:
        see: EXAMPLE

DESCRIPTION:

Bash script that copies a table from a MySQL database into a flat file,
calls Python for data processing, and writes back the result into the MySQL
database.

EXAMPLE:

1. Extract the occupations which occur with neither maximum nor minimum frequency among participants.
2. Add the column "Selected" containing Boolean values:
   "True" for occupations that match the answer in 1, and "False" otherwise.

Sample input:

+----+------+------------+
| ID | NAME | OCCUPATION |
+----+------+------------+
| 1  | Jay  | Engineer   |
| 2  | Lin  | Engineer   |
| 3  | Tom  | Architect  |
| 4  | Mat  | Engineer   |
| 5  | Kim  | Botanist   |
| 6  | Val  | Architect  |
+----+------+------------+

Sample output (adds a column after processing via Python):

+----+------+------------+----------+
| ID | NAME | OCCUPATION | SELECTED |
+----+------+------------+----------+
| 1  | Jay  | Engineer   | False    |
| 2  | Lin  | Engineer   | False    |
| 3  | Tom  | Architect  | True     |
| 4  | Mat  | Engineer   | False    |
| 5  | Kim  | Botanist   | False    |
| 6  | Val  | Architect  | True     |
+----+------+------------+----------+
'

# check if mysql server is installed
if [ -z "$(command -v mysql)" ]; then
    echo 'MySQL is not installed.'
    echo 'Install on Debian via: sudo apt-get install default-mysql-server'
    echo 'Exiting ...'
    exit 1
fi

# create database and table
sudo mysql -e "
    CREATE DATABASE IF NOT EXISTS DB;
    USE DB;

    DROP TABLE IF EXISTS participants;
    CREATE TABLE participants
    (
        ID              VARCHAR(255) NOT NULL,
        NAME            VARCHAR(255) NOT NULL,
        OCCUPATION      VARCHAR(255),
        PRIMARY KEY     (ID)
    );

    INSERT INTO participants VALUES ('1','Jay','Engineer')  ;
    INSERT INTO participants VALUES ('2','Lin','Engineer')  ;
    INSERT INTO participants VALUES ('3','Tom','Architect') ;
    INSERT INTO participants VALUES ('4','Mat','Engineer')  ;
    INSERT INTO participants VALUES ('5','Kim','Botanist')  ;
    INSERT INTO participants VALUES ('6','Val','Architect') ;
"

# query database and write into tsv file
sudo mysql -Be "SELECT * FROM DB.participants;"  > /tmp/participants.tsv

# transform file into csv
cat /tmp/participants.tsv | tr '\t' ',' > /tmp/participants.csv

# data processing via Python
cat << EOF > /tmp/filetmp1
import pandas as pd
import numpy as np

df = pd.read_csv('/tmp/participants.csv', sep=',')
values, counts = np.unique(df['OCCUPATION'], return_counts=True)
x = [values[i] for i in range(len(values)) if counts[i] not in [min(counts), max(counts)]]
print(x[0])

df['Selected'] = df['OCCUPATION'].isin(x)
df.to_csv('/tmp/filetmp2', index=False, header=False)
EOF

python3 /tmp/filetmp1

# write back into database as a new table
sudo mysql -e "
    USE DB;
    CREATE TABLE participants_updated
    (
        ID              VARCHAR(255) NOT NULL,
        NAME            VARCHAR(255),
        OCCUPATION      VARCHAR(255),
        SELECTED        VARCHAR(255)
    );

    LOAD DATA INFILE '/tmp/filetmp2' INTO TABLE participants_updated FIELDS TERMINATED BY ',';
    SELECT * FROM participants_updated;
"

# part (1) as a one-liner in Bash
cat /tmp/participants.tsv | sed '1d' | cut -f 3 | sort | uniq -c | sort -n | sed '1d' | sed '$d' | awk '{print $2}' 

rm /tmp/participants.{tsv,csv} /tmp/filetmp{1,2}
