# TokenTriage — judge replay container.
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY config ./config
COPY benchmarks ./benchmarks
COPY imgs ./imgs
RUN pip install --no-cache-dir .
# Judge replay mode is self-contained: no Ollama, no OpenRouter, no API keys.
EXPOSE 8000
CMD tokentriage serve --judge-mode --host 0.0.0.0 --port ${PORT:-8000}
