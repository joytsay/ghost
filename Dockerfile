FROM dd1:5000/workspace:pytorch1.12.1-cuda11.3-torchface1.12.0.extra
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

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update
RUN apt install gcc-9 libnccl2=2.9.9-1+cuda11.3 libnccl-dev=2.9.9-1+cuda11.3 -y
RUN apt --fix-broken install

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