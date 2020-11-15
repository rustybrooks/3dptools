FROM python:3.8

# RUN apt update -y && apt install -y libgl1-mesa-glx python3-vtk7 pyqt5-dev qt5-default

RUN mkdir -p /sr/src

COPY ./requirements.txt /srv/src/
RUN pip install -U numpy && pip install -r /srv/src/requirements.txt

COPY . ./srv/src




