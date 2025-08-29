FROM python:3.10-slim

WORKDIR /app

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

# Expose both FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Default command → Streamlit UI
CMD ["streamlit", "run", "ui.py", "--server.port=8501", "--server.address=0.0.0.0"]
