# src/generate_synthetic_data.py
"""Generate synthetic sensor readings for development/testing.
Write CSV to data/sensor_readings.csv
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
def generate_for_machine(machine_id, start_ts, minutes=60*24*30):
	"""Generate minute-level synthetic readings for a single machine.

	Args:
		machine_id: int or str identifier
		start_ts: start timestamp (datetime or string)
		minutes: number of minutes to simulate

	Returns:
		pandas.DataFrame with columns: machine_id, ts, temperature, vibration,
		pressure, current, rpm
	"""
	# minutes = number of minutes to simulate
	rng = pd.date_range(start=start_ts, periods=minutes, freq='T')
	n = len(rng)

	# base signals
	temp = 50 + 5 * np.sin(np.linspace(0, 20, n)) + np.random.normal(0, 0.5, n)
	vibration = (
		0.5 + 0.05 * np.random.randn(n) + 0.5 * np.sin(np.linspace(0, 10, n))
	)
	pressure = 10 + 0.2 * np.random.randn(n)
	current = 5 + 0.3 * np.random.randn(n)
	rpm = 1500 + 50 * np.random.randn(n)

	df = pd.DataFrame(
		{
			'machine_id': machine_id,
			'ts': rng,
			'temperature': temp,
			'vibration': vibration,
			'pressure': pressure,
			'current': current,
			'rpm': rpm,
		}
	)
	return df
if __name__ == '__main__':
	# generate for 3 machines (one week of minute-level data)
	start = datetime.now() - timedelta(days=7)
	frames = []
	for mid in [1, 2, 3]:
		frames.append(generate_for_machine(mid, start, minutes=60 * 24 * 7))
	out = pd.concat(frames, ignore_index=True)

	# Ensure output directory exists
	os.makedirs('data', exist_ok=True)
	out.to_csv('data/sensor_readings.csv', index=False)
	print('Saved data/sensor_readings.csv', out.shape)