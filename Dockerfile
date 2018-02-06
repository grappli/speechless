FROM nvidia/9.0-cudnn7-devel-ubuntu16.04

RUN sudo apt-get update && apt-get install -y --no-install-recommends \
    libav-tools \
    python3 \
    git

RUN python -m pip install --upgrade pip setuptools wheel

RUN locale-gen C.UTF-8

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV LANGUAGE=C.UTF-8

RUN pip install tensorflow
RUN mkdir /data/speechless && cd /data/speechless && git clone https://github.com/MLCogUP/speechless.git
