## Aim

Bypass the use of PL/SQL by writing a required table from mysql into a flat file, then processing the data via Python (pandas, numpy) and writing the result back into a new table in mysql server - with all calls being carried out briefly within the same Bash script.

### Task description

Given the table 'participants' below:
 
1. Extract the occupations which occur with neither maximum nor minimum frequency among participants.

2. Add the column 'Selected' containing Boolean values: 'True' for occupations that match with the answer in 1, and 'False' otherwise.

#### participants

ID | NAME | OCCUPATION
:--|:-----|:----------
1  | Jay  | Engineer
2  | Lin  | Engineer
3  | Tom  | Architect
4  | Mat  | Engineer
5  | Kim  | Botanist
6  | Val  | Architect

### Solution

Instead of querying from within mysql server and manipulating the data there, do the following in Bash (see `plsql_bypass.sh`):

0. Create a database 'DB' and a table 'participants' for the demo.
1. Extract the table from mysql (via: select * from participants) into a flat tsv file (i.e. no need for Procedural Language of PL/SQL)
2. Transform the tab-separated file into a csv file (via 'tr' in Bash).
3. Process the data as needed via Python code called from within Bash, and write the result to a csv file.
4. Write the same result into a new table in mysql server.

Step 2 gives:
```
ID,NAME,OCCUPATION
1,Jay,Engineer
2,Lin,Engineer
3,Tom,Architect
4,Mat,Engineer
5,Kim,Botanist
6,Val,Architect
```

Step 3 gives:

```
+----+------+------------+----------+
| ID | NAME | OCCUPATION | SELECTED |
+----+------+------------+----------+
| ID | NAME | OCCUPATION | Selected |
| 1  | Jay  | Engineer   | False    |
| 2  | Lin  | Engineer   | False    |
| 3  | Tom  | Architect  | True     |
| 4  | Mat  | Engineer   | False    |
| 5  | Kim  | Botanist   | False    |
| 6  | Val  | Architect  | True     |
+----+------+------------+----------+
```
