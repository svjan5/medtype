# A Dockerfile for running the MedType server.

FROM python:3.9

# Copy the source code to /opt/medtype
WORKDIR /opt/medtype
COPY . /opt/medtype

# Install gcc
RUN apt install gcc

# We need Pipenv to set up Python packages.
RUN pip install --upgrade pip
RUN pip install pipenv

WORKDIR medtype-as-service/server
# RUN pip install -r requirements.txt
RUN python setup.py install
