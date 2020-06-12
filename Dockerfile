FROM ubuntu:16.04
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt