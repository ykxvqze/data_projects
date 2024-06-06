## Aim

Export a given table from MySQL into a flat file, then process the data via Python (pandas, numpy) and write the result back into a new table in MySQL database - with all calls being carried out within a Bash script.

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

### Procedure

Instead of querying from within MySQL server and manipulating the data there, the following is carried out in Bash (see <a class="external reference" href="https://github.com/thln2ejz/data_projects/blob/master/sqlextern.sh">sqlextern.sh</a>).

0. Create a database 'DB' and a table named 'participants'.
1. Extract the table from MySQL (via: SELECT * FROM participants) into a flat tsv file
2. Transform the tab-separated file into a csv file (via 'tr' in Bash).
3. Process the data as needed via Python called from within Bash, and write the result to a csv file.
4. Write the same result into a new table in MySQL database.

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
ID,NAME,OCCUPATION,Selected
1,Jay,Engineer,False
2,Lin,Engineer,False
3,Tom,Architect,True
4,Mat,Engineer,False
5,Kim,Botanist,False
6,Val,Architect,True
```

Step 4 gives:
```
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
```
