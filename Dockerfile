FROM python:3.11-slim

RUN mkdir /app
WORKDIR /app

COPY main.py /app
COPY pyproject.toml /app
COPY entrypoint.sh /app
RUN chmod +x /app/entrypoint.sh 
RUN pip install . --no-cache 

ENTRYPOINT ["/app/entrypoint.sh"]
