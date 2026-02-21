FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY evaluate.py .
COPY setB.pth .
COPY data/test/ data/test/

CMD ["python", "evaluate.py"]
