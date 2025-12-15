from db_connect import execute_query

stmts = [
    "ALTER TABLE sensor_readings RENAME COLUMN ts TO timestamp",
    "ALTER TABLE sensor_readings RENAME COLUMN temperature TO air_temperature",
    "ALTER TABLE sensor_readings RENAME COLUMN vibration TO process_temperature",
    "ALTER TABLE sensor_readings RENAME COLUMN pressure TO torque",
    "ALTER TABLE sensor_readings RENAME COLUMN current TO rotational_speed",
    "ALTER TABLE sensor_readings RENAME COLUMN rpm TO tool_wear",
    "ALTER TABLE failure_logs ALTER COLUMN failure_date TYPE TIMESTAMP",
    "ALTER TABLE failure_logs ADD COLUMN IF NOT EXISTS failure_mode VARCHAR(255)",
    "ALTER TABLE failure_logs ADD COLUMN IF NOT EXISTS downtime_hours REAL",
]

for s in stmts:
    try:
        execute_query(s)
        print('Executed:', s)
    except Exception as e:
        print('Failed:', s, e)
