FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir uv
COPY pyproject.toml ./
COPY prorag ./prorag
COPY alembic ./alembic
COPY alembic.ini ./
COPY scripts/migrate.py ./scripts/migrate.py
COPY docker-entrypoint.sh ./
RUN uv pip install --system . && chmod +x docker-entrypoint.sh
EXPOSE 8000
ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["uvicorn", "prorag.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]
