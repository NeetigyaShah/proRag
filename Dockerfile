FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml ./
COPY prorag ./prorag
COPY alembic ./alembic
COPY alembic.ini ./
RUN uv pip install --system .
EXPOSE 8000
CMD ["uvicorn", "prorag.main:app", "--host", "0.0.0.0", "--port", "8000"]
