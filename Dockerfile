FROM nvidia/cuda:8.0-cudnn6-devel

ENV PATH=/opt/conda/bin:$PATH

# apt
RUN set -x && \
    sed -ie 's@http://archive.ubuntu.com/@http://jp.archive.ubuntu.com/@g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install --yes wget && \
    wget -q https://www.ubuntulinux.jp/ubuntu-ja-archive-keyring.gpg -O- | apt-key add - && \
    wget -q https://www.ubuntulinux.jp/ubuntu-jp-ppa-keyring.gpg -O- | apt-key add - && \
    wget https://www.ubuntulinux.jp/sources.list.d/xenial.list -O /etc/apt/sources.list.d/ubuntu-ja.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes \
        apt-transport-https \
        apt-utils \
        bash-completion \
        bc \
        bsdmainutils \
        build-essential \
        command-not-found \
        cpio \
        curl \
        debconf-i18n \
        ed \
        emacs \
        gdb \
        git \
        graphviz \
        imagemagick \
        language-pack-ja \
        libboost-all-dev \
        libopencv-dev \
        libopenmpi-dev \
        net-tools \
        openssh-server \
        screen \
        sl \
        sudo \
        unzip \
        vim \
        zip \
        zsh \
        && \
    apt-get autoremove --purge && \
    apt-get autoclean && \
    apt-get clean && \
    update-locale LANG=ja_JP.UTF-8 LANGUAGE='ja_JP:ja'

# ・sshd用ディレクトリ作成
# ・ログイン時にcudaなどのパスが通るようにしておく
# ・sudoでhttp_proxy / https_proxyが引き継がれるようにしておく
RUN set -x && \
    mkdir -pm 744 /var/run/sshd && \
    echo export PATH=$PATH:'$PATH' > /etc/profile.d/docker-env.sh && \
    echo export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:'$LD_LIBRARY_PATH' >> /etc/profile.d/docker-env.sh && \
    echo 'Defaults env_keep=http_proxy' > /etc/sudoers.d/proxy && \
    echo 'Defaults env_keep+=https_proxy' >> /etc/sudoers.d/proxy

# python
RUN set -x && \
    mkdir -p /opt/conda && \
    wget --quiet https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh && \
    echo "c989ecc8b648ab8a64731aaee9ed2e7e *Anaconda3-5.0.1-Linux-x86_64.sh" | md5sum -c - && \
    /bin/bash /Anaconda3-5.0.1-Linux-x86_64.sh -f -b -p /opt/conda && \
    rm Anaconda3-5.0.1-Linux-x86_64.sh
RUN set -x && \
    conda install --yes \
        gensim \
        graphviz \
        py-xgboost \
        && \
    conda clean --all --yes && \
    pip install --upgrade --no-cache-dir pip

# Caffe
COPY Makefile.config /opt/
RUN set -x && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes libatlas-base-dev libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev libgflags-dev libgoogle-glog-dev liblmdb-dev protobuf-compiler git wget && \
    apt-get clean && \
    ln -s /dev/null /dev/raw1394 && \
    cp -nv /opt/conda/pkgs/hdf5-*/lib/libhdf5.so.*.* /lib/x86_64-linux-gnu/ && \
    cp -nv /opt/conda/pkgs/hdf5-*/lib/libhdf5_hl.so.*.* /lib/x86_64-linux-gnu/ && \
    ldconfig && \
    git clone https://github.com/BVLC/caffe.git /opt/caffe && \
    mv /opt/Makefile.config /opt/caffe/ && \
    cd /opt/caffe && \
    make all -j4 && \
    make test -j4

# Chainer
RUN set -x && \
    LDFLAGS='-L/usr/local/nvidia/lib -L/usr/local/nvidia/lib64' pip install --no-cache-dir cupy chainer chainercv chainerrl
RUN set -x && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes libopenmpi-dev && \
    apt-get clean && \
    git clone https://github.com/NVIDIA/nccl.git /opt/nccl && \
    cd /opt/nccl && \
    make PREFIX=/usr/local install && \
    C_INCLUDE_PATH='/usr/local/cuda/targets/x86_64-linux/include' LDFLAGS='-L/usr/local/cuda/lib64/stubs' pip install --no-cache-dir chainermn

# PyTorch
RUN set -x && \
    conda install --yes pytorch torchvision cuda80 -c soumith && \
    conda clean --all --yes

# Keras+TensorFlow
RUN set -x && \
    pip install --no-cache-dir \
        git+https://www.github.com/farizrahman4u/keras-contrib.git \
        keras==2.0.9 \
        tensorflow-gpu==1.4.0

# horovod
RUN set -x && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes libopenmpi-dev && \
    apt-get clean && \
    mv /opt/conda/lib/libstdc++.so.6 /opt/conda/lib/libstdc++.so.6.bk && \
    cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/lib/ && \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64/stubs" pip install --no-cache-dir horovod && \
    rm -f /usr/local/cuda/lib64/stubs/libcuda.so.1

# その他pythonライブラリ色々
RUN set -x && \
    pip install --no-cache-dir \
        augmentor \
        better_exceptions \
        catboost \
        diskcache \
        flake8 \
        flake8-docstrings \
        flake8-pep257 \
        flask \
        flask_sqlalchemy \
        gym \
        hacking \
        hyperopt \
        janome \
        jupyterlab \
        kaggle-cli \
        opencv-python \
        prospector \
        pydot_ng \
        pytest \
        pytest-xdist \
        sklearn_pandas \
        tqdm \
        && \
    jupyter serverextension enable --py jupyterlab --sys-prefix

CMD /usr/sbin/sshd -D

