FROM nvidia/cuda:8.0-cudnn6-devel

ENV PATH=/opt/conda/bin:$PATH

# apt
RUN set -x && \
    sed -ie 's@http://archive.ubuntu.com/@http://jp.archive.ubuntu.com/@g' /etc/apt/sources.list && \
    sed -ie 's@^deb-src@# deb-src@g' /etc/apt/sources.list && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends wget software-properties-common && \
    wget -q https://www.ubuntulinux.jp/ubuntu-ja-archive-keyring.gpg -O- | apt-key add - && \
    wget -q https://www.ubuntulinux.jp/ubuntu-jp-ppa-keyring.gpg -O- | apt-key add - && \
    wget -q https://www.ubuntulinux.jp/sources.list.d/xenial.list -O /etc/apt/sources.list.d/ubuntu-ja.list && \
    add-apt-repository ppa:git-core/ppa && \
    apt-get update && \
    wget -q https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh -O- | bash && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        apt-transport-https \
        apt-utils \
        bash-completion \
        bc \
        bsdmainutils \
        ca-certificates \
        command-not-found \
        cpio \
        curl \
        debconf-i18n \
        dialog \
        ed \
        emacs \
        file \
        fonts-liberation \
        gdb \
        git \
        git-lfs \
        graphviz \
        htop \
        iftop \
        imagemagick \
        iputils-ping \
        language-pack-ja \
        less \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libnccl-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        man-db \
        net-tools \
        openssh-client \
        openssh-server \
        p7zip-full \
        protobuf-compiler \
        rsync \
        screen \
        sl \
        sudo \
        telnet \
        tmux \
        unzip \
        vim \
        zip \
        zsh \
        && \
    apt-get clean && \
    update-locale LANG=ja_JP.UTF-8 LANGUAGE='ja_JP:ja' && \
    cd /tmp && \
    git lfs install

# OpenMPI
RUN set -x && \
    wget -q https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.bz2 -O /opt/openmpi.tar.bz2 && \
    echo "757d51719efec08f9f1a7f32d58b3305 */opt/openmpi.tar.bz2" | md5sum -c - && \
    cd /opt && \
    tar xfj openmpi.tar.bz2 && \
    cd openmpi-3.0.0 && \
    ./configure --prefix=/usr/local --with-cuda --disable-mpi-fortran --disable-java && \
    make -j8 all && \
    make -j8 install && \
    ldconfig && \
    rm /opt/openmpi.tar.bz2

# python
RUN set -x && \
    mkdir -p /opt/conda && \
    wget -q https://repo.continuum.io/miniconda/Miniconda3-4.3.30-Linux-x86_64.sh -O conda.sh && \
    echo "0b80a152332a4ce5250f3c09589c7a81 *conda.sh" | md5sum -c - && \
    /bin/bash /conda.sh -f -b -p /opt/conda && \
    conda install --yes libgcc && \
    conda clean --all --yes && \
    rm conda.sh
RUN pip install --upgrade --no-cache-dir pip && \
    pip install --no-cache-dir \
        Pillow \
        bcolz \
        cython \
        gensim \
        graphviz \
        gym \
        h5py \
        matplotlib \
        mkl \
        nose \
        numba \
        numpy \
        pandas \
        pydot_ng \
        pyyaml \
        scikit-image \
        scikit-learn \
        scipy \
        six \
        sklearn_pandas \
        && \
    echo .

# Caffe
COPY Makefile.config /opt/
RUN set -x && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler libatlas-base-dev libgflags-dev libgoogle-glog-dev liblmdb-dev && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends libboost-all-dev && \
    apt-get clean && \
    ln -s /dev/null /dev/raw1394 && \
    ldconfig && \
    git clone https://github.com/BVLC/caffe.git /opt/caffe && \
    mv /opt/Makefile.config /opt/caffe/ && \
    cd /opt/caffe && \
    make -j8 all && \
    make -j8 test && \
    rm /dev/raw1394

# Chainer
RUN set -x && \
    LDFLAGS='-L/usr/local/nvidia/lib -L/usr/local/nvidia/lib64' pip install --no-cache-dir chainer chainercv chainerrl && \
    LDFLAGS='-L/usr/local/cuda/lib64/stubs' C_INCLUDE_PATH='/usr/local/cuda/targets/x86_64-linux/include' pip install --no-cache-dir chainermn

# PyTorch
RUN set -x && \
    pip install --no-cache-dir http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl && \
    pip install --no-cache-dir torchvision

# Keras+TensorFlow
RUN pip install --no-cache-dir tensorflow-gpu==1.4.1
RUN pip install --no-cache-dir keras==2.1.2

# horovod
RUN set -x && \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64/stubs" HOROVOD_GPU_ALLREDUCE=NCCL pip install --no-cache-dir horovod && \
    rm -f /usr/local/cuda/lib64/stubs/libcuda.so.1

# その他pythonライブラリ色々
RUN set -x && \
    pip install --no-cache-dir \
        git+https://www.github.com/farizrahman4u/keras-contrib.git \
        augmentor \
        better_exceptions \
        catboost \
        diskcache \
        fasteners \
        flake8 \
        flake8-docstrings \
        flake8-pep257 \
        flask \
        flask_sqlalchemy \
        hacking \
        hyperopt \
        imageio \
        janome \
        jupyterlab \
        kaggle-cli \
        opencv-python \
        prospector \
        pytest \
        pytest-xdist \
        tqdm \
        xgboost \
        && \
    jupyter serverextension enable --py jupyterlab --sys-prefix

# ・sshd用ディレクトリ作成
# ・ログイン時にcudaなどのパスが通るようにしておく
# ・sudoでhttp_proxy / https_proxyが引き継がれるようにしておく
RUN set -x && \
    mkdir -pm 744 /var/run/sshd && \
    echo export PATH=$PATH:'$PATH' > /etc/profile.d/docker-env.sh && \
    echo export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:'$LD_LIBRARY_PATH' >> /etc/profile.d/docker-env.sh && \
    echo 'Defaults env_keep += "http_proxy https_proxy ftp_proxy no_proxy"' > /etc/sudoers.d/proxy && \
    ldconfig

COPY start_sshd.sh /root/
RUN date '+%Y/%m/%d %H:%M:%S' > /image.version
CMD ["/bin/bash", "/root/start_sshd.sh"]
