INSERT INTO machines (machine_name, machine_type, location, installation_date,
manufacturer)
VALUES
('Machine A','CNC Lathe','Plant 1','2020-02-10','MakerCorp'),
('Machine B','Hydraulic Press','Plant 1','2019-07-22','PressMakers'),
('Machine C','Milling','Plant 2','2021-11-05','MillWorks');
-- failures example
INSERT INTO failure_logs (machine_id, failure_date, failure_type, description)
VALUES
(1,'2023-03-12','Bearing failure','High vibration followed by stop'),
(2,'2024-01-04','Hydraulic leak','Sudden drop in pressure');