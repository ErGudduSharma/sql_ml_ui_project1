drop database if exists test_db;

create database test_db;

use test_db;

CREATE TABLE students (
    student_id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100),
    age INT,
    gender VARCHAR(10),
    academic_performance INT,
    stress_level INT,
    sleep_quality INT,
    anxiety_level INT,
    exercise_hours FLOAT,
    study_hours FLOAT,
    social_activity INT,
    financial_stress INT,
    part_time_job VARCHAR(3),
    screen_time FLOAT,
    diet_quality INT,
    smoking INT,
    alcohol FLOAT
);


INSERT INTO students
(name, age, gender, academic_performance, stress_level, sleep_quality, anxiety_level, exercise_hours, study_hours, social_activity, financial_stress, part_time_job, screen_time, diet_quality, smoking, alcohol)
VALUES
('Alice', 20, 'Female', 4, 6, 3, 7, 2.5, 5.0, 2, 6, 'No', 6.5, 7, 0, 1.2),
('Bob', 22, 'Male', 3, 8, 2, 9, 1.0, 6.0, 1, 9, 'Yes', 8.0, 5, 3, 4.5),
('Charlie', 23, 'Male', 5, 4, 4, 5, 4.0, 7.0, 3, 5, 'No', 5.0, 8, 0, 0.5),
('David', 21, 'Male', 2, 9, 2, 8, 0.5, 4.5, 0, 10, 'Yes', 9.5, 4, 5, 6.0),
('Eva', 24, 'Female', 5, 3, 4, 4, 5.0, 7.5, 4, 4, 'No', 4.5, 9, 0, 0.0),
('Fiona', 19, 'Female', 3, 7, 3, 8, 2.0, 6.0, 2, 7, 'Yes', 7.0, 6, 1, 2.0),
('George', 22, 'Male', 4, 5, 3, 6, 3.0, 6.5, 3, 6, 'No', 6.0, 7, 0, 1.0),
('Hannah', 23, 'Female', 2, 8, 2, 9, 1.5, 5.5, 1, 8, 'Yes', 8.5, 5, 2, 3.5),
('Ian', 20, 'Male', 4, 6, 3, 7, 2.5, 6.0, 2, 7, 'No', 7.0, 7, 0, 0.8),
('Julia', 21, 'Female', 5, 4, 4, 5, 4.5, 7.0, 3, 5, 'No', 5.5, 8, 0, 0.0),
('Kevin', 22, 'Male', 3, 7, 2, 8, 1.2, 6.5, 2, 9, 'Yes', 9.0, 6, 3, 4.2),
('Laura', 24, 'Female', 5, 3, 5, 4, 5.5, 8.0, 4, 4, 'No', 4.0, 9, 0, 0.0),
('Michael', 23, 'Male', 2, 9, 2, 9, 0.8, 4.0, 1, 10, 'Yes', 10.0, 5, 4, 5.5),
('Nina', 20, 'Female', 4, 6, 3, 7, 2.0, 6.5, 2, 6, 'No', 6.5, 7, 0, 1.0),
('Oscar', 21, 'Male', 3, 8, 2, 9, 1.5, 5.5, 1, 8, 'Yes', 8.0, 5, 2, 3.0),
('Paula', 19, 'Female', 5, 4, 4, 5, 4.2, 7.5, 3, 5, 'No', 5.0, 8, 0, 0.0),
('Quinn', 22, 'Other', 3, 7, 3, 8, 2.5, 6.0, 2, 7, 'Yes', 7.5, 6, 1, 2.5),
('Raj', 23, 'Male', 2, 9, 2, 9, 0.9, 4.5, 1, 9, 'Yes', 9.5, 5, 3, 4.8),
('Sara', 20, 'Female', 4, 6, 3, 7, 2.8, 6.0, 3, 6, 'No', 6.0, 7, 0, 0.5),
('Tom', 24, 'Male', 5, 3, 5, 4, 5.0, 8.0, 4, 4, 'No', 4.0, 9, 0, 0.0);


select * from students;