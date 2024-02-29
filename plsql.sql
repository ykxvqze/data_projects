--
-- Database tables
--

/*
SELECT * FROM employee;

+--------+------------+-----------+------------+--------+--------+----------+-----------+
| emp_id | first_name | last_name | birth_day  | gender | salary | super_id | branch_id |
+--------+------------+-----------+------------+--------+--------+----------+-----------+
|    100 | Dave       | Doe       | 1968-11-19 | M      | 250000 |     NULL |         1 |
|    101 | John       | Smith     | 1960-08-10 | F      | 110000 |      100 |         1 |
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

--
-- PL/SQL
--

-- show employee or client
CREATE OR REPLACE PROCEDURE viewPerson(auxID NUMBER)
IS
    auxEmployee     employee.emp_id%TYPE;
    auxClient       client.client_id%TYPE;
    auxFN           employee.first_name%TYPE;
    auxLN           employee.last_name%TYPE;
    auxDOB          employee.birth_day%TYPE;
    auxGender       employee.gender%TYPE;
    auxSalary       employee.salary%TYPE;
    auxSuperID      employee.super_id%TYPE;
    auxBranchID     employee.branch_id%TYPE;
    auxBranchName   branch.branch_name%TYPE;
    auxClientID     client.client_id%TYPE;
    auxClientName   client.client_name%TYPE;
    auxTotalSales   relations.total_sales%TYPE;

BEGIN
    SELECT COUNT(*) INTO auxEmployeeCount
    FROM employee
    WHERE emp_id = auxID; 

    SELECT COUNT(*) INTO auxClientCount
    FROM client
    WHERE client_id = auxID;

    IF auxEmployeeCount > 0 THEN
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

        ELSIF auxClientCount > 0 THEN
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
    auxEmployeeCount     NUMBER;
    auxClientCount       NUMBER;

BEGIN
    SELECT COUNT(*) INTO auxEmployeeCount
    FROM employee
    WHERE emp_id = auxEmployeeID; 

    SELECT COUNT(*) INTO auxClientCount
    FROM client
    WHERE client_id = auxClientID;

    IF auxEmployeeCount > 0 AND auxClientCount > 0 THEN
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

    auxEmployeeCount     NUMBER;
    auxBranchIDCount     NUMBER;

BEGIN
    SELECT COUNT(*) INTO auxEmployeeCount
    FROM employee
    WHERE emp_id = auxEmployeeID; 

    IF auxEmployeeCount = 0 THEN
        IF auxSuperID IS NULL OR EXISTS (
            SELECT emp_id
            FROM employee
            WHERE emp_id = auxSuperID
        ) THEN

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
    auxFN             VARCHAR2(20);
    auxLN             VARCHAR2(20);
    auxDOB            VARCHAR2(10);
    auxGender         VARCHAR2(1);
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
    auxClientCount      NUMBER;
    auxBranchIDCount    NUMBER;
BEGIN
    SELECT COUNT(*) INTO auxClientCount
    FROM client
    WHERE client_id = auxClientID; 

    IF auxClientCount = 0 THEN
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
    auxClientID     NUMBER;
    auxClientName   VARCHAR2(20);
    auxBranchID     NUMBER;
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

   auxBranchIDCount    NUMBER;

BEGIN
    SELECT COUNT(*) INTO auxBranchIDCount
    FROM branch
    WHERE branch_id = auxBranchID;

    IF auxBranchIDCount > 0 THEN
        DBMS_OUTPUT.PUT_LINE('Branch ID already exists.');
    ELSE
        IF auxMngID IS NULL OR EXISTS (
            SELECT mng_id
            FROM branch
            WHERE mng_id = auxMngID
        ) THEN

            INSERT INTO branch VALUES (auxBranchID, auxBranchName, auxMngID, auxMngStartDate);
        ELSE
            DBMS_OUTPUT.PUT_LINE('Manager ID is invalid.');
        END IF;
    END IF;
END;
/

SET SERVEROUTPUT ON;
DECLARE
    auxBranchID     branch.branch_id%TYPE,
    auxBranchName   branch.branch_name%TYPE,
    auxMngID        branch.mng_id%TYPE,
    auxMngStartDate branch.mng_start_date%TYPE,
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
    auxBranchID        employee.branch_id%TYPE
    )
IS

    auxEmployeeIDCount  NUMBER;
    auxBranchIDCount    NUMBER;

BEGIN
    SELECT COUNT(*) INTO auxEmployeeIDCount
    FROM employee
    WHERE emp_id = auxEmployeeID;

    IF auxEmployeeIDCount > 0 THEN
        IF auxSuperID IS NULL OR EXISTS (
            SELECT emp_id
            FROM employee
            WHERE emp_id = auxSuperID
        ) THEN
            SELECT 
        
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

   auxBranchIDCount    NUMBER;

BEGIN
    SELECT COUNT(*) INTO auxBranchIDCount
    FROM branch
    WHERE branch_id = auxBranchID;

    IF auxBranchIDCount = 0 THEN
        DBMS_OUTPUT.PUT_LINE('Branch ID does not exist to update.');
    ELSE
        IF auxMngID IS NULL OR EXISTS (
            SELECT mng_id
            FROM branch
            WHERE mng_id = auxMngID
        ) THEN

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
    auxBranchID     branch.branch_id%TYPE,
    auxBranchName   branch.branch_name%TYPE,
    auxMngID        branch.mng_id%TYPE,
    auxMngStartDate branch.mng_start_date%TYPE,
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

        DBMS_OUTPUT.PUT_LINE('First Name: ' || vFirstName || ', Last Name: ' || vLastName);
        
    END LOOP;
    CLOSE c_employee;
END;
/

SET SERVEROUTPUT ON;
listEmployeeNames;
