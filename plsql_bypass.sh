#!/usr/bin/env bash
: ' 
A demo for bypassing PL/SQL via Bash/Python

USAGE:  ./plsql_bypass.sh

OUTPUT:
        see: EXAMPLE

DESCRIPTION:

* Bash script that writes a table from mysql into a flat file, calls
Python for data processing, and writes back the result into mysql server.
This bypasses the need to use Procedural Language or customized SQL
queries within the database server (for non-Big data).

EXAMPLE:

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

Sample output (adding a column after processing via Python; see "plsql_bypass.md"):

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

J.A., xrzfyvqk_k1jw@pm.me
'

# install mysql server if not already present
[ `command -v mysql` ] || sudo apt-get install default-mysql-server

# create database and table
sudo mysql -Be "
create database DB;
use DB;

create table participants
(
ID varchar(255) not null,
NAME varchar(255) not null,
OCCUPATION varchar(255),
primary key (ID)
);

insert into participants values ('1','Jay','Engineer');
insert into participants values ('2','Lin','Engineer');
insert into participants values ('3','Tom','Architect');
insert into participants values ('4','Mat','Engineer');
insert into participants values ('5','Kim','Botanist');
insert into participants values ('6','Val','Architect');
"

# query mysql and write into tsv file
sudo mysql -Be "select * from DB.participants;"  > participants.tsv  # option -B for tab-separated, i.e. w/o table borders

# transform file into csv
cat participants.tsv | tr '\t' ',' > participants.csv  # csv file

# data processing via Python
cat << EOF > filetmp1
import pandas as pd
import numpy as np

df = pd.read_csv('./participants.csv', sep=',')
values, counts = np.unique(df['OCCUPATION'], return_counts=True)
x = [values[i] for i in range(len(values)) if counts[i] not in [min(counts), max(counts)]]
print(x[0])

df['Selected'] = df['OCCUPATION'].isin(x)
df.to_csv('/tmp/filetmp2', index=False, header=False)
EOF

python3 ./filetmp1

# write back into mysql as a new table
sudo mysql -Be "
use DB;
create table participants_updated
(
ID varchar(255) not null,
NAME varchar(255),
OCCUPATION varchar(255),
SELECTED varchar(255)
);

load data infile '/tmp/filetmp2' into table participants_updated fields terminated by ',';
select * from participants_updated;
"

# answer to part (1) as a one-liner in Bash
cat participants.tsv | sed '1d' | cut -f 3 | sort | uniq -c | sort -n | sed '1d' | sed '$d' | tr -s ' ' | cut -d ' ' -f 3

rm participants.tsv participants.csv ./filetmp1 /tmp/filetmp2 
