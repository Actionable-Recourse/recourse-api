FROM python:3.8-slim-buster

LABEL maintainer="Takashi Yabuta"

RUN apt-get update -y && \
    apt-get install -y python-pip python-dev

WORKDIR /

COPY . .

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt


# CMD [ "python3", "-m" , "flask", "run", "--host=https://recourse-api.herokuapp.com/"]

#Run the command
CMD gunicorn --bind 0.0.0.0:$PORT app:app