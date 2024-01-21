FROM dexxpl33988012.xxp.sap.corp:11000/icn-ml/tf12-py2-3-cuda8

MAINTAINER Alexey Medvedev <alexey.medvedev@sap.com>

ADD tf_unet /root/tf_unet
#ADD car_train_128 /root/car_train_128
ADD unet_car.py /root/unet_car.py

#ADD requirements.txt /root/

# Uncomment this if you have special python requirements
#RUN cd /root && \
#    pip install -r requirements.txt

WORKDIR "/root"

CMD ["python", "unet_car.py"]