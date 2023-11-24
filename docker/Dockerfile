FROM dd1:5000/workspace:pytorch2.0.1-cuda11.8-torchface1.12.0.1.extra
USER root
WORKDIR /workspace/
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y \
        sudo \
        wget \
        vim \
        python3.8 \
        python3.8-distutils \
        ffmpeg \
        libsm6 \
        libxext6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt-get update
RUN apt install libnccl2 libnccl-dev -y --allow-change-held-packages

RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.8 get-pip.py
COPY ./ /workspace/
RUN chmod 777 -R /root/miniconda
RUN pip install -r requirements_latest.txt

ARG USERNAME
ARG USERID
ARG GROUPID

RUN echo ${USERNAME}:${USERNAME}
RUN echo ${USERID}:${USERID}
RUN echo ${GROUPID}:${GROUPID}

RUN useradd -m ${USERNAME} -u ${USERID} -g ${GROUPID} -G sudo -s /bin/sh
RUN echo ${USERNAME}:${USERNAME} | chpasswd  && adduser ${USERNAME} sudo

USER ${USERNAME}
WORKDIR /code/ghost
# ENTRYPOINT ["tail", "-f", "/dev/null"]