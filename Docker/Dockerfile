#FROM nvcr.io/nvidia/pytorch:22.02-py3
FROM nvcr.io/nvidia/pytorch:22.10-py3
ARG DEBIAN_FRONTEND=noninteractive

#Install os Requirements

#ADD os_requirements.txt .
#RUN apt update -y && apt install -y $(cat os_requirements.txt|grep -v '#')

# Install Python packages (from requirements.txt):

ADD requirements.txt .
RUN pip install -r requirements.txt

#Taken from https://u.group/thinking/how-to-put-jupyter-notebooks-in-a-dockerfile/ on how to enable jupyter on docker
RUN pip install jupyter 
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
RUN pip install pytorchvideo

ARG UNAME=user
ARG GID=1000
ARG UID=1000
RUN groupadd -g $GID $UNAME
RUN useradd -m -u $UID -g $GID -s /bin/bash $UNAME

RUN chmod -R 777 /workspace
RUN chmod -R 777 /home
USER $UNAME