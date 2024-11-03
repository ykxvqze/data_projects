--
-- Database tables
--

USE db;

SELECT * FROM employee;
/*
+--------+------------+-----------+------------+--------+--------+----------+-----------+
| emp_id | first_name | last_name | birth_day  | gender | salary | super_id | branch_id |
+--------+------------+-----------+------------+--------+--------+----------+-----------+
|    100 | Dave       | Doe       | 1968-11-19 | M      | 250000 |     NULL |         1 |
|    101 | John       | Smith     | 1960-08-10 | M      | 110000 |      100 |         1 |
|    102 | Pete       | Marley    | 1966-04-11 | M      |  75000 |      100 |         2 |
|    103 | Angel      | Martin    | 1972-05-28 | F      |  63000 |      102 |         2 |
|    104 | Valery     | Kap       | 1981-01-03 | F      |  55000 |      102 |         2 |
|    105 | Tom        | Taylor    | 1956-01-11 | M      |  69000 |      102 |         2 |
|    106 | Lee        | Porter    | 1968-08-09 | M      |  78000 |      100 |         3 |
|    107 | Andy       | Bern      | 1972-06-21 | M      |  65000 |      106 |         3 |
|    108 | Jim        | Tal       | 1979-11-05 | M      |  71000 |      106 |         3 |
|    109 | Oscar      | Martin    | 1969-03-19 | M      |  69000 |      106 |         3 |
|    110 | Eric       | Malon     | 1979-03-18 | M      |  69000 |      106 |         3 |
|    111 | Pam        | Bert      | 1989-01-17 | F      |  69000 |      106 |         3 |
+--------+------------+-----------+------------+--------+--------+----------+-----------+
*/

SELECT * FROM branch;
/*
+-----------+-------------+--------+----------------+
| branch_id | branch_name | mng_id | mng_start_date |
+-----------+-------------+--------+----------------+
|         1 | Corporate-1 |    100 | 2006-02-09     |
|         2 | Corporate-2 |    102 | 1992-04-06     |
|         3 | Corporate-3 |    106 | 1998-02-13     |
|         4 | Corporate-4 |   NULL | NULL           |
+-----------+-------------+--------+----------------+
*/

SELECT * FROM client;
/*
+-----------+-------------+-----------+
| client_id | client_name | branch_id |
+-----------+-------------+-----------+
|       400 | Client-A    |         2 |
|       401 | Client-B    |         2 |
|       402 | Client-C    |         3 |
|       403 | Client-D    |         3 |
|       404 | Client-E    |         2 |
|       405 | Client-F    |         3 |
|       406 | Client-G    |         2 |
+-----------+-------------+-----------+
*/

SELECT * FROM relations;
/*
+--------+-----------+-------------+
| emp_id | client_id | total_sales |
+--------+-----------+-------------+
|    102 |       401 |      267000 |
|    102 |       406 |       15000 |
|    105 |       400 |       55000 |
|    105 |       404 |       33000 |
|    105 |       406 |      130000 |
|    107 |       403 |        5000 |
|    107 |       405 |       26000 |
|    108 |       402 |       22500 |
|    108 |       403 |       12000 |
+--------+-----------+-------------+
*/

--
-- Queries
--

-- Names of employees with overall sales exceeding 50000
SELECT first_name, last_name
FROM employee
WHERE emp_id IN (
                 SELECT emp_id
                 FROM ( 
                       SELECT emp_id, SUM(total_sales) AS totals
                       FROM relations
                       GROUP BY emp_id
                      ) AS temp_table
                 WHERE totals > 50000
                );
/*
+------------+-----------+
| first_name | last_name |
+------------+-----------+
| Pete       | Marley    |
| Tom        | Taylor    |
+------------+-----------+
*/

-- IDs and names of clients who deal with the branch whose manager is Pete Marley
SELECT client_id, client_name
FROM client
WHERE branch_id = (
                   SELECT branch_id
                   FROM branch
                   WHERE mng_id = (
                                   SELECT emp_id
                                   FROM employee
                                   WHERE first_name = 'Pete' AND last_name ='Marley'
                                   LIMIT 1
                                  )
                  );
/*
+-----------+-------------+
| client_id | client_name |
+-----------+-------------+
|       400 | Client-A    |
|       401 | Client-B    |
|       404 | Client-E    |
|       406 | Client-G    |
+-----------+-------------+
*/

-- Names of employees with relations to clients who deal with the Corporate-2 branch
SELECT first_name, last_name
FROM employee
WHERE emp_id IN ( 
                 SELECT emp_id
                 FROM relations
                 WHERE client_id IN (
                                     SELECT client_id
                                     FROM client
                                     WHERE branch_id = ( 
                                                        SELECT branch_id
                                                        FROM branch
                                                        WHERE branch_name = 'Corporate-2'
                                                       )
                                     )
                );
/*
+------------+-----------+
| first_name | last_name |
+------------+-----------+
| Pete       | Marley    |
| Tom        | Taylor    |
+------------+-----------+
*/

-- Clients with an expenditure exceeding 100000
SELECT client_name
FROM client
WHERE client_id IN (
                    SELECT temp_table.client_id
                    FROM (
                          SELECT client_id, SUM(total_sales) as total
                          FROM relations
                          GROUP BY client_id
                          HAVING total > 100000
                         ) AS temp_table
                   );
/*
+-------------+
| client_name |
+-------------+
| Client-B    |
| Client-G    |
+-------------+
*/
