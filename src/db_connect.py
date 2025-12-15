# src/db_connect.py
"""
Database connection helper for PostgreSQL using psycopg2.
This file provides a connect() function that returns (conn, cur)
Remember to call conn.commit() after write operations and close both.
"""
import os
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd

# Prefer centralized configuration if available
try:
	from config_file import Config
except Exception:
	Config = None

# Build DB config from environment variables with fallbacks to Config (if present)
DB_CFG = {
	'host': os.getenv('DB_HOST', getattr(Config, 'DB_HOST', 'localhost') if Config else 'localhost'),
	'dbname': os.getenv('DB_NAME', getattr(Config, 'DB_NAME', 'zepto_sql_project') if Config else 'zepto_sql_project'),
	'user': os.getenv('DB_USER', getattr(Config, 'DB_USER', 'postgres') if Config else 'postgres'),
	'password': os.getenv('DB_PASSWORD', getattr(Config, 'DB_PASSWORD', '') if Config else ''),
	'port': int(os.getenv('DB_PORT', getattr(Config, 'DB_PORT', 5432) if Config else 5432))
}
def connect():
	"""Create and return a new (conn, cur) tuple using DB_CFG."""
	conn = psycopg2.connect(**DB_CFG)
	cur = conn.cursor()
	return conn, cur
def close(conn, cur):
	"""Close cursor and connection if they are provided.

	Args:
		conn: psycopg2 connection or None
		cur: psycopg2 cursor or None
	"""
	if cur is not None:
		try:
			cur.close()
		except Exception:
			pass
	if conn is not None:
		try:
			conn.close()
		except Exception:
			pass


if __name__ == '__main__':
	"""Simple runtime check: attempt to connect and query Postgres version.


	This allows running `python src/db_connect.py` inside a virtualenv to
	verify connectivity. It reads DB credentials from environment variables
	if set (DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT).
	"""
	import sys

	conn = None
	cur = None
	try:
		conn, cur = connect()
		cur.execute('SELECT version();')
		ver = cur.fetchone()
		print('Connected to database. Postgres version:', ver)
		sys.exit(0)
	except Exception as e:
		print('Database connection failed:', e)
		sys.exit(1)
	finally:
		try:
			close(conn, cur)
		except Exception:
			pass

def execute_query(sql, params=None, fetch=False):
	"""Execute SQL. If fetch=True return rows, else return None.

	Automatically opens and closes a connection.
	"""
	conn, cur = None, None
	try:
		conn, cur = connect()
		cur.execute(sql, params or ())
		if fetch:
			rows = cur.fetchall()
			return rows
		else:
			conn.commit()
			return None
	finally:
		try:
			close(conn, cur)
		except Exception:
			pass


def fetch_dataframe(sql, params=None):
	"""Run a query and return a pandas.DataFrame.

	Returns empty DataFrame if no rows.
	"""
	conn, cur = None, None
	try:
		conn, cur = connect()
		cur.execute(sql, params or ())
		rows = cur.fetchall()
		cols = [desc[0] for desc in cur.description] if cur.description else []
		df = pd.DataFrame(rows, columns=cols)
		return df
	finally:
		try:
			close(conn, cur)
		except Exception:
			pass


def bulk_insert(table, columns, data, page_size=1000):
	"""Insert many rows into `table` using psycopg2.extras.execute_values.

	Args:
		table: table name (str)
		columns: list of column names
		data: iterable of tuples
		page_size: batch size for execute_values
	"""
	if not data:
		return
	conn, cur = None, None
	try:
		conn, cur = connect()
		cols = ','.join(columns)
		sql = f"INSERT INTO {table} ({cols}) VALUES %s"
		execute_values(cur, sql, data, page_size=page_size)
		conn.commit()
	finally:
		try:
			close(conn, cur)
		except Exception:
			pass