FROM nvidia/cuda:9.0-cudnn7-devel

ENV PATH=/opt/conda/bin:$PATH

# 実行時に残さないようにENVではなくARGでnoninteractive
ARG DEBIAN_FRONTEND=noninteractive

# apt用プロキシ(apt-cacher-ng用)
ARG APT_PROXY=$http_proxy

# apt
RUN set -x && \
    sed -ie 's@http://archive.ubuntu.com/ubuntu/@http://ftp.riken.go.jp/Linux/ubuntu/@g' /etc/apt/sources.list && \
    sed -ie 's@^deb-src@# deb-src@g' /etc/apt/sources.list && \
    http_proxy=$APT_PROXY apt-get update && \
    http_proxy=$APT_PROXY apt-get install --yes --no-install-recommends wget curl software-properties-common apt-utils && \
    wget -q https://www.ubuntulinux.jp/ubuntu-ja-archive-keyring.gpg -O- | apt-key add - && \
    wget -q https://www.ubuntulinux.jp/ubuntu-jp-ppa-keyring.gpg -O- | apt-key add - && \
    wget -q https://www.ubuntulinux.jp/sources.list.d/xenial.list -O /etc/apt/sources.list.d/ubuntu-ja.list && \
    add-apt-repository ppa:git-core/ppa && \
    wget -q https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh -O- | bash && \
    http_proxy=$APT_PROXY apt-get update && \
    http_proxy=$APT_PROXY apt-get install --yes --no-install-recommends \
        apt-file \
        apt-transport-https \
        bash-completion \
        bc \
        bsdmainutils \
        build-essential \
        ca-certificates \
        cifs-utils \
        cmake \
        command-not-found \
        cpio \
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
        inetutils-traceroute \
        iotop \
        iproute2 \
        iputils-ping \
        language-pack-ja \
        less \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libjpeg-dev \
        libleveldb-dev \
        liblmdb-dev \
        libnccl-dev \
        libopencv-dev \
        libpng-dev \
        libprotobuf-dev \
        libsnappy-dev \
        man-db \
        net-tools \
        openssh-client \
        openssh-server \
        p7zip-full \
        protobuf-compiler \
        psmisc \
        rsync \
        screen \
        sl \
        smbclient \
        sudo \
        swig \
        telnet \
        tmux \
        tmuxinator \
        unzip \
        valgrind \
        vim \
        whiptail \
        zip \
        zsh \
        && \
    apt-get clean && \
    update-locale LANG=ja_JP.UTF-8 LANGUAGE='ja_JP:ja' && \
    cd /tmp && \
    git lfs install

# OpenMPI
# 参考：https://github.com/uber/horovod/blob/master/Dockerfile
RUN set -x && \
    wget -q https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.bz2 -O /opt/openmpi.tar.bz2 && \
    echo "757d51719efec08f9f1a7f32d58b3305 */opt/openmpi.tar.bz2" | md5sum -c - && \
    cd /opt && \
    tar xfj openmpi.tar.bz2 && \
    cd openmpi-3.0.0 && \
    ./configure --prefix=/usr/local --with-cuda --disable-mpi-fortran --disable-java --enable-orterun-prefix-by-default && \
    make -j$(nproc) all && \
    make -j$(nproc) install && \
    ldconfig && \
    rm /opt/openmpi.tar.bz2 && \
    echo "hwloc_base_binding_policy = none" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "rmaps_base_mapping_policy = slot" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo "btl_tcp_if_exclude = lo,docker0" >> /usr/local/etc/openmpi-mca-params.conf && \
    echo NCCL_DEBUG=INFO >> /etc/nccl.conf && \
    echo NCCL_SOCKET_IFNAME=^docker0 >> /etc/nccl.conf

# devpi-server用
ARG PIP_PROXY=$http_proxy
ARG PIP_TRUSTED_HOST=""
ARG PIP_INDEX_URL=""

# python
RUN set -x && \
    mkdir -p /opt/conda && \
    wget -q https://repo.continuum.io/miniconda/Miniconda3-4.3.30-Linux-x86_64.sh -O conda.sh && \
    echo "0b80a152332a4ce5250f3c09589c7a81 *conda.sh" | md5sum -c - && \
    /bin/bash /conda.sh -f -b -p /opt/conda && \
    conda install --yes libgcc && \
    conda clean --all --yes && \
    rm conda.sh
RUN http_proxy=$PIP_PROXY pip install --upgrade --no-cache-dir pip && \
    http_proxy=$PIP_PROXY pip install --no-cache-dir \
        Pillow \
        bcolz \
        cython \
        futures==3.1.1 \
        gensim \
        graphviz \
        gym \
        h5py \
        matplotlib \
        mkl \
        mpi4py \
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
RUN set -x && \
    git clone --branch=master --single-branch --depth=1 https://github.com/BVLC/caffe.git /opt/caffe
COPY Makefile.config /opt/caffe/
RUN set -x && \
    ln -s /dev/null /dev/raw1394 && \
    cd /opt/caffe && \
    ldconfig && \
    make -j$(nproc) all pycaffe && \
    echo "/opt/caffe/build/lib" >> /etc/ld.so.conf.d/caffe.conf && \
    ldconfig && \
    rm /dev/raw1394

# Chainer
RUN set -x && \
    http_proxy=$PIP_PROXY pip install --no-cache-dir cupy chainer chainercv chainerrl chainermn

# PyTorch
RUN set -x && \
    pip install --no-cache-dir http://download.pytorch.org/whl/cu90/torch-0.3.0.post4-cp36-cp36m-linux_x86_64.whl && \
    http_proxy=$PIP_PROXY pip install --no-cache-dir torchvision

# Keras+TensorFlow
RUN http_proxy=$PIP_PROXY pip install --no-cache-dir tensorflow-gpu==1.5.0
RUN http_proxy=$PIP_PROXY pip install --no-cache-dir keras==2.1.3

# horovod
RUN set -x && \
    ldconfig /usr/local/cuda/lib64/stubs && \
    http_proxy=$PIP_PROXY pip install --no-cache-dir horovod && \
    ldconfig

# その他pythonライブラリ色々
RUN set -x && \
    http_proxy=$PIP_PROXY pip install --no-cache-dir \
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
        lightgbm \
        opencv-python \
        prospector \
        pytest \
        pytest-timeout \
        pytest-xdist \
        tqdm \
        xgboost \
        && \
    jupyter serverextension enable --py jupyterlab --sys-prefix

# ・sshd用ディレクトリ作成
# ・cuda、python、caffeなどのパスを通す
# ・matplotlibがエラーにならないようにMPLBACKEND=Aggを設定
# ・sudoでhttp_proxyなどが引き継がれるようにしておく
# ・sudoで/opt/conda/binにパスが通っているようにしておく
RUN set -x && \
    mkdir -pm 744 /var/run/sshd && \
    echo 'export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/opt/conda/bin:/opt/caffe/build/tools:$PATH' > /etc/profile.d/docker.sh && \
    echo 'export CAFFE_ROOT=/opt/caffe' >> /etc/profile.d/docker.sh && \
    echo 'export PYTHONPATH=/opt/caffe/python' >> /etc/profile.d/docker.sh && \
    echo 'export MPLBACKEND=Agg' >> /etc/profile.d/docker.sh && \
    echo 'Defaults env_keep += "http_proxy https_proxy ftp_proxy no_proxy"' > /etc/sudoers.d/docker && \
    echo 'Defaults secure_path = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin:/opt/conda/bin"' >> /etc/sudoers.d/docker && \
    echo 'Defaults always_set_home' >> /etc/sudoers.d/docker && \
    chmod 0440 /etc/sudoers.d/* && \
    visudo --check && \
    ldconfig

# monkey patch
COPY sitecustomize.py /opt/conda/lib/python3.6/site-packages/

COPY start_sshd.sh /root/
RUN date '+%Y/%m/%d %H:%M:%S' > /image.version
CMD ["/bin/bash", "/root/start_sshd.sh"]
