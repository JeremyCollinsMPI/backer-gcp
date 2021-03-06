
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
RUN git clone https://github.com/huggingface/transformers.git
WORKDIR transformers
RUN pip install -e .
RUN pip install requests 
RUN pip install flask
RUN pip install flask-cors
RUN pip install Werkzeug
RUN pip install gunicorn
RUN pip install sentencepiece protobuf

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
RUN python download_models.py
CMD exec gunicorn --bind :$PORTFROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True


RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
RUN git clone https://github.com/huggingface/transformers.git
WORKDIR transformers
RUN pip install -e .
RUN pip install requests 
RUN pip install flask
RUN pip install flask-cors
RUN pip install Werkzeug
RUN pip install gunicorn
# RUN pip install sentencepiece protobuf
RUN pip install pandas
RUN pip install sklearn
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt




# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
ENV TFHUB_CACHE_DIR=/app/tfhub_modules
# COPY . ./

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
# Timeout is set to 0 to disable the timeouts of the workers to allow Cloud Run to handle instance scaling.
# RUN python download_models.py
# CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app --workers 1 --threads 8 --timeout 0 main:app


