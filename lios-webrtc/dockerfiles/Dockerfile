FROM ghcr.io/prefix-dev/pixi:latest

# Accept username as build argument
ARG USER=ubuntu

# Switch to root to modify apt sources and install packages
USER root

RUN rm -rf /etc/apt/sources.list.d/ubuntu.sources
COPY tsinghua-apt-noble /etc/apt/sources.list

# Update package lists and install apt-file
RUN apt-get update && \
  apt-get install -y --no-install-recommends apt-file zsh kitty-terminfo sudo

RUN apt-get update && apt-get install -y --no-install-recommends \
  rsync python3 python3-pip \
  git curl build-essential pkg-config ca-certificates \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-libav \
  gstreamer1.0-nice \
  gstreamer1.0-libav libgstrtspserver-1.0-dev libges-1.0-dev libssl-dev \
  gir1.2-gst-plugins-bad-1.0 \
  python3-gi python3-websockets

# Rename ubuntu user to the specified USER
RUN usermod -l ${USER} ubuntu && \
  groupmod -n ${USER} ubuntu && \
  usermod -d /home/${USER} -m ${USER} && \
  echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN apt-file update
# Switch back to the renamed user
USER ${USER}

# Set working directory
ENV TERM=xterm-256color 
WORKDIR /home/${USER}
ENTRYPOINT [ "zsh" ]
