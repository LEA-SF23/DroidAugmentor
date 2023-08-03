FROM tensorflow/tensorflow:2.13.0
RUN apt-get update 
RUN apt-get -y install vim wget python3 python3-pip
WORKDIR /droidaugmentor
COPY ./requirements.txt /droidaugmentor/
COPY ./scripts/docker_install_requirements.sh /droidaugmentor/
RUN bash /droidaugmentor/docker_install_requirements.sh
RUN bash
