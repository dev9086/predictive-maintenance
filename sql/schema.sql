CREATE TABLE machines (
machine_id SERIAL PRIMARY KEY,
machine_name VARCHAR(100),
machine_type VARCHAR(50),
location VARCHAR(100),
installation_date DATE,
manufacturer VARCHAR(100)
);
CREATE TABLE sensor_readings (
reading_id BIGSERIAL PRIMARY KEY,
machine_id INT REFERENCES machines(machine_id) ON DELETE CASCADE,
	timestamp TIMESTAMP NOT NULL,
	air_temperature REAL,
	process_temperature REAL,
	rotational_speed INT,
	torque REAL,
	tool_wear INT
);
CREATE TABLE failure_logs (
failure_id SERIAL PRIMARY KEY,
machine_id INT REFERENCES machines(machine_id) ON DELETE CASCADE,
	failure_date TIMESTAMP,
	failure_type VARCHAR(100),
	failure_mode VARCHAR(255),
	description TEXT,
	downtime_hours REAL
);
CREATE TABLE external_data (
external_id SERIAL PRIMARY KEY,
machine_id INT NULL,
source VARCHAR(255),
key VARCHAR(100),
value TEXT,
scraped_at TIMESTAMP DEFAULT now()
);
-- Optional: table to store model predictions
CREATE TABLE model_predictions (
pred_id SERIAL PRIMARY KEY,
machine_id INT REFERENCES machines(machine_id),
ts TIMESTAMP DEFAULT now(),
failure_prob REAL,
predicted_rul REAL,
model_version VARCHAR(50)
);