FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

# change home directory
WORKDIR /home

# install deps
ADD install-deps /home/install-deps
RUN ./install-deps

WORKDIR /home/style-transfer

ADD requirements.txt /home/style-transfer/requirement.txt
RUN pip3 install -r requirement.txt

ADD . /home/style-transfer

