FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports
EXPOSE 8000 8501

# Run both services
CMD sh -c 'python src/init_db.py && (uvicorn src.fastapi_server:app --host 0.0.0.0 --port 8000 &) && streamlit run src/streamlit_dashboard.py --server.port 8501 --server.address 0.0.0.0'
