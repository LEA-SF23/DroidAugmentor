FROM sf23/droidaugmentor:latest
RUN apt-get update 
RUN apt-get -y install wget net-tools vim
RUN pip install pipenv
WORKDIR /droidaugmentor
USER root
RUN rm -rf /droidaugmentor/*
COPY ./ /droidaugmentor/
#RUN pipenv install -r /droidaugmentor/requirements.txt
RUN mv /droidaugmentor/scripts/run_app_in_docker.sh /usr/bin/docker_run.sh
RUN chmod +x /usr/bin/docker_run.sh 
CMD ["docker_run.sh"]
