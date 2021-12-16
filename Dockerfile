# A Dockerfile for running the MedType server.

FROM conda/miniconda3

# Update conda
RUN conda update conda python

# Install gcc and g++
RUN conda install -c conda-forge gcc
RUN conda install -c conda-forge gxx

# Install rust
RUN conda install -c anaconda rust-nightly 

# We need pip to set up Python packages.
RUN pip install --upgrade pip

# Copy the source code to /opt/medtype
WORKDIR /opt/medtype
COPY . /opt/medtype

# Get working on medtype-as-service
WORKDIR medtype-as-service

# There is a "./resources" directory that contains downloaded resources
# such as models. To avoid redownloading them every time we run setup.py,
# we should store them in a separate volume.
VOLUME resources

# Setup MedType as service
RUN bash setup.sh
#RUN pip install -r requirements.txt
#RUN python setup.py install
