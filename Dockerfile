# set base image (host OS)
FROM python:3.8.10

# set the working directory in the container
ADD . /app
WORKDIR /app

# copy the dependencies file to the
RUN python -m venv venv
RUN venv/bin/pip install --upgrade pip


# install dependencies
RUN venv/bin/pip install --no-cache  sklearn pandas flask flask_restful

# copy the content of the local src directory to the working directory
ADD src/  .
EXPOSE 5000
# command to run on container start
CMD . venv/bin/activate && exec python script.py

