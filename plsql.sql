--
-- database tables
--

/*
SELECT * FROM employee;

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

/*
SELECT * FROM branch;

+-----------+-------------+--------+----------------+
| branch_id | branch_name | mng_id | mng_start_date |
+-----------+-------------+--------+----------------+
|         1 | Corporate-1 |    100 | 2006-02-09     |
|         2 | Corporate-2 |    102 | 1992-04-06     |
|         3 | Corporate-3 |    106 | 1998-02-13     |
|         4 | Corporate-4 |   NULL | NULL           |
+-----------+-------------+--------+----------------+
*/

/*
SELECT * FROM client;

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

/*
SELECT * FROM relations;

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

---
--- create initial tables
---

CREATE TABLE employee (
    emp_id      INT PRIMARY KEY,
    first_name  VARCHAR2(40),
    last_name   VARCHAR2(40),
    birth_day   DATE,
    gender      VARCHAR2(1),
    salary      INT,
    super_id    INT,
    branch_id   INT
);

CREATE TABLE branch (
    branch_id       INT PRIMARY KEY,
    branch_name     VARCHAR2(40),
    mng_id          INT,
    mng_start_date  DATE
);

CREATE TABLE client (
  client_id     INT PRIMARY KEY,
  client_name   VARCHAR2(40),
  branch_id     INT
);

CREATE TABLE relations (
  emp_id        INT,
  client_id     INT,
  total_sales   INT,
  PRIMARY KEY(emp_id, client_id)
  );

---
--- insert data
---

INSERT INTO employee VALUES(100, 'Dave', 'Doe', TO_DATE('1968-11-19','YYYY-MM-DD'), 'M', 250000, NULL, 1);
INSERT INTO employee VALUES(101, 'John', 'Smith', TO_DATE('1960-08-10','YYYY-MM-DD'), 'M', 110000, 100, 1);
INSERT INTO employee VALUES(102, 'Pete', 'Marley', TO_DATE('1966-04-11','YYYY-MM-DD'), 'M', 75000, 100, 2);
INSERT INTO employee VALUES(103, 'Angel', 'Martin', TO_DATE('1972-05-28','YYYY-MM-DD'), 'F', 63000, 102, 2);
INSERT INTO employee VALUES(104, 'Valery', 'Kap', TO_DATE('1981-01-03','YYYY-MM-DD'), 'F', 55000, 102, 2);
INSERT INTO employee VALUES(105, 'Tom', 'Taylor', TO_DATE('1956-01-11','YYYY-MM-DD'), 'M', 69000, 102, 2);
INSERT INTO employee VALUES(106, 'Lee', 'Porter', TO_DATE('1968-08-09','YYYY-MM-DD'), 'M', 78000, 100, 3);
INSERT INTO employee VALUES(107, 'Andy', 'Bern', TO_DATE('1972-06-21','YYYY-MM-DD'), 'M', 65000, 106, 3);
INSERT INTO employee VALUES(108, 'Jim', 'Tal', TO_DATE('1979-11-05','YYYY-MM-DD'), 'M', 71000, 106, 3);
INSERT INTO employee VALUES(109, 'Oscar', 'Martin', TO_DATE('1969-03-19','YYYY-MM-DD'), 'M', 69000, 106, 3);
INSERT INTO employee VALUES(110, 'Eric', 'Malon', TO_DATE('1979-03-18','YYYY-MM-DD'), 'M', 69000, 106, 3);
INSERT INTO employee VALUES(111, 'Pam', 'Bert', TO_DATE('1989-01-17','YYYY-MM-DD'), 'F', 69000, 106, 3);

INSERT INTO branch VALUES(1, 'Corporate-1', 100, TO_DATE('2006-02-09','YYYY-MM-DD'));
INSERT INTO branch VALUES(2, 'Corporate-2', 102, TO_DATE('1992-04-06','YYYY-MM-DD'));
INSERT INTO branch VALUES(3, 'Corporate-3', 106, TO_DATE('1998-02-13','YYYY-MM-DD'));
INSERT INTO branch VALUES(4, 'Corporate-4', NULL, NULL);

INSERT INTO client VALUES(400, 'Client-A', 2);
INSERT INTO client VALUES(401, 'Client-B', 2);
INSERT INTO client VALUES(402, 'Client-C', 3);
INSERT INTO client VALUES(403, 'Client-D', 3);
INSERT INTO client VALUES(404, 'Client-E', 2);
INSERT INTO client VALUES(405, 'Client-F', 3);
INSERT INTO client VALUES(406, 'Client-G', 2);

INSERT INTO relations VALUES(102, 401, 267000);
INSERT INTO relations VALUES(102, 406, 15000);
INSERT INTO relations VALUES(105, 400, 55000);
INSERT INTO relations VALUES(105, 404, 33000);
INSERT INTO relations VALUES(105, 406, 130000);
INSERT INTO relations VALUES(107, 403, 5000);
INSERT INTO relations VALUES(107, 405, 26000);
INSERT INTO relations VALUES(108, 402, 22500);
INSERT INTO relations VALUES(108, 403, 12000);

---
--- set constraints
---

ALTER TABLE employee
ADD FOREIGN KEY(branch_id)
REFERENCES branch(branch_id)
ON DELETE SET NULL;

ALTER TABLE employee
ADD FOREIGN KEY(super_id)
REFERENCES employee(emp_id)
ON DELETE SET NULL;

ALTER TABLE branch
ADD FOREIGN KEY(mng_id)
REFERENCES employee(emp_id)
ON DELETE SET NULL;

ALTER TABLE client
ADD FOREIGN KEY(branch_id)
REFERENCES branch(branch_id)
ON DELETE SET NULL;

ALTER TABLE relations
ADD FOREIGN KEY(emp_id)
REFERENCES employee(emp_id)
ON DELETE CASCADE;

ALTER TABLE relations
ADD FOREIGN KEY(client_id)
REFERENCES client(client_id)
ON DELETE CASCADE;

--
-- PL/SQL
--

-- show employee or client
CREATE OR REPLACE PROCEDURE viewPerson(auxID NUMBER)
IS
    auxEmployee         employee.emp_id%TYPE;
    auxClient           client.client_id%TYPE;
    auxFN               employee.first_name%TYPE;
    auxLN               employee.last_name%TYPE;
    auxDOB              employee.birth_day%TYPE;
    auxGender           employee.gender%TYPE;
    auxSalary           employee.salary%TYPE;
    auxSuperID          employee.super_id%TYPE;
    auxBranchID         employee.branch_id%TYPE;
    auxBranchName       branch.branch_name%TYPE;
    auxClientID         client.client_id%TYPE;
    auxClientName       client.client_name%TYPE;
    auxTotalSales       relations.total_sales%TYPE;
    auxEmployeeIDCount  NUMBER;
    auxClientIDCount    NUMBER;

BEGIN
    SELECT COUNT(*) INTO auxEmployeeIDCount
    FROM employee
    WHERE emp_id = auxID; 

    SELECT COUNT(*) INTO auxClientIDCount
    FROM client
    WHERE client_id = auxID;

    IF auxEmployeeIDCount > 0 THEN
        SELECT first_name, last_name, birth_day, gender, salary, super_id, branch_id
        INTO auxFN, auxLN, auxDOB, auxGender, auxSalary, auxSuperID, auxBranchID
        FROM employee
        WHERE emp_id = auxID;

        SELECT branch_name INTO auxBranchName
        FROM branch
        WHERE branch_id = auxBranchID;

        DBMS_OUTPUT.PUT_LINE('EMPLOYEE WITH ID ' || auxID || ' INFO');
        DBMS_OUTPUT.PUT_LINE('-------------------------------------');
        DBMS_OUTPUT.PUT_LINE('FIRST NAME  : ' || auxFN);
        DBMS_OUTPUT.PUT_LINE('LAST NAME   : ' || auxLN);
        DBMS_OUTPUT.PUT_LINE('BIRTH DAY   : ' || auxDOB);
        DBMS_OUTPUT.PUT_LINE('GENDER      : ' || auxGender);
        DBMS_OUTPUT.PUT_LINE('SALARY      : ' || auxSalary);
        DBMS_OUTPUT.PUT_LINE('SUPER ID    : ' || auxSuperID);
        DBMS_OUTPUT.PUT_LINE('BRANCH ID   : ' || auxBranchID);
        DBMS_OUTPUT.PUT_LINE('BRANCH NAME : ' || auxBranchName);
        DBMS_OUTPUT.PUT_LINE('-------------------------------------');

        BEGIN 
            SELECT client_id, total_sales INTO auxClientID, auxTotalSales
            FROM relations
            WHERE emp_id = auxID;

            DBMS_OUTPUT.PUT_LINE('TOTAL SALES : ' || auxTotalSales);

            EXCEPTION
              WHEN NO_DATA_FOUND THEN
                DBMS_OUTPUT.PUT_LINE('Employee does not handle any clients.');
        END;

        BEGIN
            SELECT client_name INTO auxClientName
            FROM client
            WHERE client_id = auxClientID;

            DBMS_OUTPUT.PUT_LINE('CLIENT NAME : ' || auxClientName);

            EXCEPTION
              WHEN NO_DATA_FOUND THEN
                DBMS_OUTPUT.PUT_LINE('Employee does not handle any clients.');
        END;

        ELSIF auxClientIDCount > 0 THEN
            SELECT client_name, branch_id
            INTO auxClientName, auxBranchID
            FROM client
            WHERE client_id = auxID;

            SELECT branch_name INTO auxBranchName
            FROM branch
            WHERE branch_id = auxBranchID;

            DBMS_OUTPUT.PUT_LINE('CLIENT WITH ID ' || auxID || ' INFO');
            DBMS_OUTPUT.PUT_LINE('-------------------------------------');
            DBMS_OUTPUT.PUT_LINE('CLIENT NAME : ' || auxClientName);
            DBMS_OUTPUT.PUT_LINE('BRANCH ID   : ' || auxBranchID);
            DBMS_OUTPUT.PUT_LINE('BRANCH NAME : ' || auxBranchName);
            DBMS_OUTPUT.PUT_LINE('-------------------------------------');

        ELSE
            DBMS_OUTPUT.PUT_LINE('Not a valid ID.');
      END IF;
END;
/

SET SERVEROUTPUT ON;
DECLARE
    auxID     NUMBER;
BEGIN
    auxID := &personID;
    viewPerson(auxID);
END;
/

-- insert into relations
CREATE OR REPLACE PROCEDURE addSales(auxEmployeeID NUMBER, auxClientID NUMBER, auxSales NUMBER)
IS
    auxEmployeeIDCount     NUMBER;
    auxClientIDCount       NUMBER;

BEGIN
    SELECT COUNT(*) INTO auxEmployeeIDCount
    FROM employee
    WHERE emp_id = auxEmployeeID; 

    SELECT COUNT(*) INTO auxClientIDCount
    FROM client
    WHERE client_id = auxClientID;

    IF auxEmployeeIDCount > 0 AND auxClientIDCount > 0 THEN
        INSERT INTO relations VALUES (auxEmployeeID, auxClientID, auxSales);
    ELSE
        DBMS_OUTPUT.PUT_LINE('Employee ID or Client ID does not exist.');
    END IF;
END;
/

SET SERVEROUTPUT ON;
DECLARE
    auxEmployeeID     NUMBER;
    auxClientID       NUMBER;
    auxSales          NUMBER;
BEGIN
    auxEmployeeID := &EmployeeID;
    auxClientID   := &ClientID;
    auxSales      := &Sales;
    addSales(auxEmployeeID, auxClientID, auxSales);
END;
/

-- add employee
CREATE OR REPLACE PROCEDURE addEmployee(
    auxEmployeeID IN NUMBER,
    auxFN IN VARCHAR2,
    auxLN IN VARCHAR2,
    auxDOB IN VARCHAR2,
    auxGender IN VARCHAR2,
    auxSalary IN NUMBER,
    auxSuperID in NUMBER,
    auxBranchID IN NUMBER
    )
IS

    auxEmployeeIDCount    NUMBER;
    auxBranchIDCount      NUMBER;
    auxSuperIDCount       NUMBER;

BEGIN
    SELECT COUNT(*) INTO auxEmployeeIDCount
    FROM employee
    WHERE emp_id = auxEmployeeID; 

    IF auxEmployeeIDCount = 0 THEN
        SELECT COUNT(*) INTO auxSuperIDCount
        FROM employee
        WHERE emp_id = auxSuperID;

        IF auxSuperIDCount > 0 OR auxSuperID IS NULL THEN
            SELECT COUNT(*) INTO auxBranchIDCount
            FROM branch
            WHERE branch_id = auxBranchID;

            IF auxBranchIDCount > 0 THEN
                INSERT INTO employee VALUES (
                    auxEmployeeID,
                    auxFN,
                    auxLN,
                    TO_DATE(auxDOB,'YYYY-MM-DD'),
                    auxGender,
                    auxSalary,
                    auxSuperID,
                    auxBranchID
                );
                DBMS_OUTPUT.PUT_LINE('Employee added successfully.');
            ELSE
                DBMS_OUTPUT.PUT_LINE('Branch ID does not exist.');
            END IF;
        ELSE
            DBMS_OUTPUT.PUT_LINE('Supervisor ID is invalid.');
        END IF;
    ELSE
        DBMS_OUTPUT.PUT_LINE('Employee ID already exists.');
    END IF;
END;
/

SET SERVEROUTPUT ON;
DECLARE
    auxEmployeeID     NUMBER;
    auxFN             employee.first_name%TYPE;
    auxLN             employee.last_name%TYPE;
    auxDOB            employee.birth_day%TYPE;
    auxGender         employee.gender%TYPE;
    auxSalary         NUMBER;
    auxSuperID        NUMBER;
    auxBranchID       NUMBER;
BEGIN
    auxEmployeeID := &EmployeeID;
    auxFN := '&FirstName';
    auxLN := '&LastName';
    auxDOB := '&DOB';
    auxGender := '&Gender';
    auxSalary := &Salary;
    auxSuperID := &SupervisorID;
    auxBranchID := &BranchID;
    addEmployee(auxEmployeeID, auxFN, auxLN, auxDOB, auxGender, auxSalary, auxSuperID, auxBranchID);
END;
/

-- add client
CREATE OR REPLACE PROCEDURE addClient(
    auxClientID IN NUMBER,
    auxClientName IN VARCHAR2,
    auxBranchID IN NUMBER
    )
IS
    auxClientIDCount      NUMBER;
    auxBranchIDCount    NUMBER;
BEGIN
    SELECT COUNT(*) INTO auxClientIDCount
    FROM client
    WHERE client_id = auxClientID; 

    IF auxClientIDCount = 0 THEN
        SELECT COUNT(*) INTO auxBranchIDCount
        FROM branch
        WHERE branch_id = auxBranchID;

        IF auxBranchIDCount > 0 THEN              
            INSERT INTO client VALUES (
                auxClientID,
                auxClientName,
                auxBranchID
            );
            DBMS_OUTPUT.PUT_LINE('Client added successfully.');
        ELSE
            DBMS_OUTPUT.PUT_LINE('Branch ID does not exist.');
        END IF;
    ELSE
        DBMS_OUTPUT.PUT_LINE('Client ID already exists.');
    END IF;
END;
/

SET SERVEROUTPUT ON;
DECLARE
    auxClientID     client.client_id%TYPE;
    auxClientName   client.client_name%TYPE;
    auxBranchID     client.branch_id%TYPE;
BEGIN
    auxClientID := &ClientID;
    auxClientName := '&ClientName';
    auxBranchID := &BranchID;
    addClient(auxClientID, auxClientName, auxBranchID);
END;
/

-- insert into branch
CREATE OR REPLACE PROCEDURE addBranch(
    auxBranchID     branch.branch_id%TYPE,
    auxBranchName   branch.branch_name%TYPE,
    auxMngID        branch.mng_id%TYPE,
    auxMngStartDate branch.mng_start_date%TYPE
    )
IS

   auxBranchIDCount NUMBER;
   auxMngIDCount    NUMBER;

BEGIN
    SELECT COUNT(*) INTO auxBranchIDCount
    FROM branch
    WHERE branch_id = auxBranchID;

    IF auxBranchIDCount > 0 THEN
        DBMS_OUTPUT.PUT_LINE('Branch ID already exists.');
    ELSE
        SELECT COUNT(*) INTO auxMngIDCount
        FROM employee
        WHERE emp_id = auxMngID;

        IF auxMngID IS NULL OR auxMngIDCount > 0 THEN
            INSERT INTO branch VALUES (auxBranchID, auxBranchName, auxMngID, auxMngStartDate);
        ELSE
            DBMS_OUTPUT.PUT_LINE('Manager ID is invalid.');
        END IF;
    END IF;
END;
/

SET SERVEROUTPUT ON;
DECLARE
    auxBranchID     branch.branch_id%TYPE;
    auxBranchName   branch.branch_name%TYPE;
    auxMngID        branch.mng_id%TYPE;
    auxMngStartDate branch.mng_start_date%TYPE;
BEGIN
    auxBranchID := &branch_id;
    auxBranchName := '&branch_name';
    auxMngID := &mng_id;
    auxMngStartDate := '&mng_start_date';
    addBranch(auxBranchID, auxBranchName, auxMngID, auxMngStartDate);
END;
/

-- update info on employee
CREATE OR REPLACE PROCEDURE updateEmployee(
    auxEmployeeID   employee.emp_id%TYPE,
    auxFN           employee.first_name%TYPE,
    auxLN           employee.last_name%TYPE,
    auxDOB          employee.birth_day%TYPE,
    auxGender       employee.gender%TYPE,
    auxSalary       employee.salary%TYPE,
    auxSuperID      employee.super_id%TYPE,
    auxBranchID     employee.branch_id%TYPE
    )
IS

    auxEmployeeIDCount  NUMBER;
    auxBranchIDCount    NUMBER;
    auxSuperIDCount    NUMBER;

BEGIN
    SELECT COUNT(*) INTO auxEmployeeIDCount
    FROM employee
    WHERE emp_id = auxEmployeeID;

    IF auxEmployeeIDCount > 0 THEN
        SELECT COUNT(*) INTO auxSuperIDCount
        FROM employee
        WHERE emp_id = auxSuperID;

        IF auxSuperID IS NULL OR auxSuperIDCount > 0 THEN      
            SELECT COUNT(*) INTO auxBranchIDCount
            FROM branch
            WHERE branch_id = auxBranchID;

            IF auxBranchIDCount > 0 THEN
                UPDATE employee
                SET first_name = auxFN, last_name = auxLN, birth_day = auxDOB, gender = auxGender, salary = auxSalary, super_id = auxSuperID, branch_id = auxBranchID
                WHERE emp_id = auxEmployeeID; 
            ELSE
                DBMS_OUTPUT.PUT_LINE('Branch ID does not exist.');
            END IF;
        ELSE
            DBMS_OUTPUT.PUT_LINE('Supervisor ID is invalid.');
        END IF;
    ELSE
        DBMS_OUTPUT.PUT_LINE('Employee ID does not exist.');
    END IF;
END;
/

SET SERVEROUTPUT ON;
DECLARE
    auxEmployeeID   employee.emp_id%TYPE;
    auxFN           employee.first_name%TYPE;
    auxLN           employee.last_name%TYPE;
    auxDOB          employee.birth_day%TYPE;
    auxGender       employee.gender%TYPE;
    auxSalary       employee.salary%TYPE;
    auxSuperID      employee.super_id%TYPE;
    auxBranchID     employee.branch_id%TYPE;
BEGIN
    auxEmployeeID := &employeeID; 
    auxFN := '&employee_first_name';
    auxLN := '&employee_last_name';
    auxDOB := '&employee_date_of_birth';
    auxGender := '&employee.gender';
    auxSalary := &employee_salary;
    auxSuperID := &supervisor_id;
    auxBranchID := &employee_branch_id;
    updateEmployee(auxEmployeeID, auxFN, auxLN, auxDOB, auxGender, auxSalary, auxSuperID, auxBranchID);
END;
/
    
-- update info on client
CREATE OR REPLACE PROCEDURE updateClient(
    auxClientID     client.client_id%TYPE,
    auxClientName   client.client_name%TYPE,
    auxBranchID     client.branch_id%TYPE
    )
IS

    auxClientIDCount  NUMBER;
    auxBranchIDCount  NUMBER;

BEGIN
    SELECT COUNT(*) INTO auxClientIDCount
    FROM client
    WHERE client_id = auxClientID;

    IF auxClientIDCount > 0 THEN
        SELECT COUNT(*) INTO auxBranchIDCount
        FROM branch
        WHERE branch_id = auxBranchID;

        IF auxBranchIDCount > 0 THEN
            UPDATE client
            SET client_name = auxClientName, branch_id = auxBranchID
            WHERE client_id = auxClientID; 
        ELSE
            DBMS_OUTPUT.PUT_LINE('Branch ID does not exist.');
        END IF;
    ELSE
        DBMS_OUTPUT.PUT_LINE('Client ID does not exist.');
    END IF;
END;
/

SET SERVEROUTPUT ON;
DECLARE
    auxClientID     client.client_id%TYPE;
    auxClientName   client.client_name%TYPE;
    auxBranchID     client.branch_id%TYPE;
BEGIN
    auxClientID := &clientID; 
    auxClientName := '&client_name';
    auxBranchID := &client_branch_id;
    updateClient(auxClientID, auxClientName, auxBranchID);
END;
/


-- update total_sales in relations table
CREATE OR REPLACE PROCEDURE updateSales(
    auxEmployeeID   relations.emp_id%TYPE,
    auxClientID     relations.client_id%TYPE,
    auxSales        relations.total_sales%TYPE
    )
IS

    auxRowCount  NUMBER;

BEGIN
    SELECT COUNT(*) INTO auxRowCount
    FROM relations
    WHERE emp_id = auxEmployeeID AND client_id = auxClientID;

    IF auxRowCount > 0 THEN
        UPDATE relations
        SET total_sales = auxSales
        WHERE emp_id = auxEmployeeID AND client_id = auxClientID; 
    ELSE
        DBMS_OUTPUT.PUT_LINE('Employee/Client pair does not exist.');
    END IF;
END;
/

SET SERVEROUTPUT ON;
DECLARE
    auxEmployeeID   relations.emp_id%TYPE;
    auxClientID     relations.client_id%TYPE;
    auxSales        relations.total_sales%TYPE;
BEGIN
    auxEmployeeID := &employeeID;
    auxClientID := &clientID;
    auxSales := &total_sales;
    updateSales(auxEmployeeID, auxClientID, auxSales);
END;
/

-- update branch
CREATE OR REPLACE PROCEDURE updateBranch(
    auxBranchID     branch.branch_id%TYPE,
    auxBranchName   branch.branch_name%TYPE,
    auxMngID        branch.mng_id%TYPE,
    auxMngStartDate branch.mng_start_date%TYPE
    )
IS

   auxBranchIDCount   NUMBER;
   auxMngIDCount      NUMBER;

BEGIN
    SELECT COUNT(*) INTO auxBranchIDCount
    FROM branch
    WHERE branch_id = auxBranchID;

    IF auxBranchIDCount = 0 THEN
        DBMS_OUTPUT.PUT_LINE('Branch ID does not exist to update.');
    ELSE
        SELECT COUNT(*) INTO auxMngIDCount
        FROM employee
        WHERE emp_id = auxMngID;

        IF auxMngID IS NULL OR auxMngIDCount > 0 THEN
            UPDATE branch
            SET branch_name = auxBranchName, mng_id = auxMngID, mng_start_date = auxMngStartDate
            WHERE branch_id = auxBranchID;
        ELSE
            DBMS_OUTPUT.PUT_LINE('Manager ID is invalid.');
        END IF;
    END IF;
END;
/

SET SERVEROUTPUT ON;
DECLARE
    auxBranchID     branch.branch_id%TYPE;
    auxBranchName   branch.branch_name%TYPE;
    auxMngID        branch.mng_id%TYPE;
    auxMngStartDate branch.mng_start_date%TYPE;
BEGIN
    auxBranchID := &branch_id;
    auxBranchName := '&branch_name';
    auxMngID := &mng_id;
    auxMngStartDate := '&mng_start_date';
    updateBranch(auxBranchID, auxBranchName, auxMngID, auxMngStartDate);
END;
/

-- list employees first name and last name
CREATE OR REPLACE PROCEDURE listEmployeeNames
IS
    CURSOR c_employee
    IS
        SELECT first_name, last_name
        FROM employee;

    vFirstName  employee.first_name%TYPE;
    vLastNAme   employee.last_name%TYPE;

BEGIN
    OPEN c_employee;
    LOOP
        FETCH c_employee INTO vFirstName, vLastName;
        EXIT WHEN c_employee%NOTFOUND;

        DBMS_OUTPUT.PUT_LINE(vFirstName || ' ' || vLastName);
        
    END LOOP;
    CLOSE c_employee;
END;
/

SET SERVEROUTPUT ON;
BEGIN
    listEmployeeNames;
END;
/
