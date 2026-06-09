FROM python:3.11-slim
RUN pip install --no-cache-dir mlflow==2.13.0
EXPOSE 5000
VOLUME ["/mlflow"]
CMD ["mlflow", "server", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--default-artifact-root", "/mlflow/artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5000"]
