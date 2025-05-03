FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

COPY . /app/

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]