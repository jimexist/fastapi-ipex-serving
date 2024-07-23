FROM intel/intel-extension-for-pytorch:2.3.0-pip-base

WORKDIR /opt/fastapi_ipex_serving/

COPY ./requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "fastapi_ipex_serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
