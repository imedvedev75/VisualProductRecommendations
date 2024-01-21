FROM dexxpl33988012.xxp.sap.corp:11000/icn-ml/tf12-py2-3-cuda8

MAINTAINER Alexey Medvedev <alexey.medvedev@sap.com>

ADD model /root/model
ADD train.py /root/train.py
ADD params.py /root/params.py

ADD requirements.txt /root/

# Uncomment this if you have special python requirements
RUN cd /root && \
    pip install -r requirements.txt

WORKDIR "/root"

CMD ["python", "train.py"]