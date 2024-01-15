FROM ubuntu:22.04

WORKDIR /

RUN apt-get update && apt-get install -y \
    gstreamer1.0-plugins-base \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    python3-pip \
    libcairo2-dev \
    pkg-config \
    python3-dev \
    libgirepository1.0-dev \
    python3-gst-1.0 \
    && rm -rf /var/lib/apt/lists/*

COPY src/ .

RUN pip3 install -r requirements.txt

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]

