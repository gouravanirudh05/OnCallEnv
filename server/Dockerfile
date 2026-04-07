FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml /app/pyproject.toml
COPY env /app/env
COPY data /app/data
COPY server /app/server
COPY README.md /app/README.md
COPY openenv.yaml /app/openenv.yaml

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

EXPOSE 8000

CMD ["python", "-m", "server.app"]
