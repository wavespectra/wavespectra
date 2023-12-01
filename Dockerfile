FROM  ubuntu:22.04

LABEL maintainer "Rafael Guedes <r.guedes@oceanum.science>"
LABEL org.opencontainers.image.source=https://github.com/wavespectra/wavespectra

RUN echo "--------------- Installing system libraries ---------------" && \
    apt update && apt upgrade -y && \
    DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt -y install \
        gfortran \
        python3-pip && \
    apt clean all

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

RUN echo "-------------------- Installing wavespectra --------------------" && \
    pip install ipython wavespectra[extra]

RUN pip install ipython