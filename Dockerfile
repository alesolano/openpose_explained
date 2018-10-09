FROM ubuntu:16.04

WORKDIR /root

# Add requirements and TensorFlow wheel
COPY requirements.txt /root
RUN mkdir /root/tensorflow
COPY tensorflow/tensorflow-1.6.0rc1-cp27-cp27mu-manylinux1_x86_64.whl /root/tensorflow

# Install Python packages (TensorFlow, OpenCV...)
RUN apt-get update
RUN apt-get install -y python-pip && \
    pip install --upgrade pip==9.0.3 && \
    pip install -r requirements.txt && \
    pip install tensorflow/tensorflow-1.6.0rc1-cp27-cp27mu-manylinux1_x86_64.whl

# Install ROS
RUN apt-get install -y lsb-release
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116
RUN apt-get update
RUN apt-get install -y ros-kinetic-ros-base
RUN apt-get install -y ros-kinetic-vision-opencv
RUN rosdep init	&& rosdep update
RUN echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc

# Jupyter notebooks
COPY OpenPoseVGGvsMobile.ipynb /root
COPY Node_Camera.ipynb /root
COPY Node_OpenPose.ipynb /root
COPY Node_Window.ipynb /root
COPY estimator.py /root
COPY helper.py /root

# Test image
RUN mkdir /root/data
COPY data/test.jpg /root/data

# TensorFlow models
RUN mkdir /root/models
COPY models/openpose_mobile_opt.pb /root/models
RUN apt-get install -y wget
RUN wget https://www.dropbox.com/s/zxv2kw9rxrfgeht/openpose_vgg_opt.pb -P /root/models

EXPOSE 8888



