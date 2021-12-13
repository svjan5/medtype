# A Dockerfile for running the MedType server.

FROM python:3.9

# We need Pipenv to set up Python packages.
RUN pip install pipenv
RUN pipenv install

# Set up medtype-as-service
RUN medtype-as-service/setup.sh
