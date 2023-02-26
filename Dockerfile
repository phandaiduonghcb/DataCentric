FROM pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.11-cuda11.3.1
WORKDIR /DataCentric
COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y