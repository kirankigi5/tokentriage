# TokenTriage — deployable as a single container (Cloud Run compatible).
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY config ./config
COPY benchmarks ./benchmarks
RUN pip install --no-cache-dir .
# Keys arrive via env at deploy time — NEVER baked into the image.
EXPOSE 8000
CMD ["tokentriage", "serve", "--host", "0.0.0.0", "--port", "8000"]
