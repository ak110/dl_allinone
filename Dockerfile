FROM nvidia/cuda:9.0-cudnn7-devel

# 実行時に残さないようにENVではなくARGでnoninteractive
ARG DEBIAN_FRONTEND=noninteractive

# apt用プロキシ(apt-cacher-ng用)
ARG APT_PROXY=$http_proxy

# apt
RUN set -x && \
    sed -ie 's@http://archive.ubuntu.com/ubuntu/@http://ftp.riken.go.jp/Linux/ubuntu/@g' /etc/apt/sources.list && \
    sed -ie 's@^deb-src@# deb-src@g' /etc/apt/sources.list && \
    http_proxy=$APT_PROXY apt-get update && \
    http_proxy=$APT_PROXY apt-get install --yes --no-install-recommends \
        apt-transport-https \
        apt-utils \
        build-essential \
        ca-certificates \
        curl \
        software-properties-common \
        wget \
        && \
    wget -q https://www.ubuntulinux.jp/ubuntu-ja-archive-keyring.gpg -O- | apt-key add - && \
    wget -q https://www.ubuntulinux.jp/ubuntu-jp-ppa-keyring.gpg -O- | apt-key add - && \
    wget -q https://www.ubuntulinux.jp/sources.list.d/xenial.list -O /etc/apt/sources.list.d/ubuntu-ja.list && \
    add-apt-repository ppa:git-core/ppa && \
    wget -q https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh -O- | bash && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# https://github.com/uber/horovod/blob/master/Dockerfile
ENV CUDNN_VERSION=7.0.5.15-1+cuda9.0
ENV NCCL_VERSION=2.2.12-1+cuda9.0

# aptその2
RUN set -x && \
    http_proxy=$APT_PROXY apt-get update && \
    http_proxy=$APT_PROXY apt-get install --yes --no-install-recommends --allow-downgrades \
        apt-file \
        automake \
        bash-completion \
        bc \
        bsdmainutils \
        cifs-utils \
        cmake \
        command-not-found \
        cpio \
        debconf-i18n \
        dialog \
        dpkg-dev \
        ed \
        emacs \
        file \
        fonts-ipafont \
        fonts-liberation \
        g++ \
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
        libbz2-dev \
        libcudnn7=$CUDNN_VERSION \
        libgdbm-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libjpeg-dev \
        libleveldb-dev \
        liblmdb-dev \
        liblzma-dev \
        libnccl-dev=$NCCL_VERSION \
        libnccl2=$NCCL_VERSION \
        libncurses5-dev \
        libopencv-dev \
        libpng-dev \
        libprotobuf-dev \
        libreadline-dev \
        libsnappy-dev \
        libsqlite3-dev \
        libssl-dev \
        libtool \
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
        subversion \
        sudo \
        swig \
        tcl-dev \
        telnet \
        tk-dev \
        tmux \
        tmuxinator \
        unzip \
        valgrind \
        vim \
        whiptail \
        zip \
        zlib1g-dev \
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
    rm /opt/openmpi.tar.bz2

# python
# https://github.com/docker-library/python/blob/master/3.6/stretch/Dockerfile
ARG GPG_KEY="0D96DF4D4110E5C43FBFB17F2D347EA6AA65421D"
ARG PYTHON_VERSION="3.6.6"
RUN set -ex \
	&& wget -O python.tar.xz "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz" \
	&& wget -O python.tar.xz.asc "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz.asc" \
	&& export GNUPGHOME="$(mktemp -d)" \
	&& gpg --keyserver ha.pool.sks-keyservers.net --recv-keys "$GPG_KEY" \
	&& gpg --batch --verify python.tar.xz.asc python.tar.xz \
	&& { command -v gpgconf > /dev/null && gpgconf --kill all || :; } \
	&& rm -rf "$GNUPGHOME" python.tar.xz.asc \
	&& mkdir -p /usr/src/python \
	&& tar -xJC /usr/src/python --strip-components=1 -f python.tar.xz \
	&& rm python.tar.xz \
	\
	&& cd /usr/src/python \
	&& gnuArch="$(dpkg-architecture --query DEB_BUILD_GNU_TYPE)" \
	&& ./configure \
		--build="$gnuArch" \
		--enable-loadable-sqlite-extensions \
		--enable-shared \
		--with-system-expat \
		--with-system-ffi \
		--without-ensurepip \
	&& make -j "$(nproc)" \
	&& make install \
	&& ldconfig \
	\
	&& find /usr/local -depth \
		\( \
			\( -type d -a \( -name test -o -name tests \) \) \
			-o \
			\( -type f -a \( -name '*.pyc' -o -name '*.pyo' \) \) \
		\) -exec rm -rf '{}' + \
	&& rm -rf /usr/src/python \
    && cd /usr/local/bin \
    && ln -s idle3 idle \
    && ln -s pydoc3 pydoc \
    && ln -s python3 python \
    && ln -s python3-config python-config

# devpi-server用
ARG PIP_PROXY=$http_proxy
ARG PIP_TRUSTED_HOST=""
ARG PIP_INDEX_URL=""

# pip
RUN set -x && \
    wget 'https://bootstrap.pypa.io/get-pip.py' -O get-pip.py && \
    python get-pip.py --no-cache-dir && \
    rm -f get-pip.py
RUN set -x && \
    http_proxy=$PIP_PROXY pip install --upgrade --no-cache-dir pip && \
    http_proxy=$PIP_PROXY pip install --no-cache-dir \
        cython \
        setuptools_scm \
        numpy \
        && \
    http_proxy=$PIP_PROXY pip install --no-cache-dir \
        Pillow \
        bcolz \
        fastrlock \
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
        pandas \
        pbr \
        pydot \
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
    http_proxy=$PIP_PROXY pip install --no-cache-dir cupy-cuda90 chainer chainercv chainerrl chainermn

# PyTorch
ARG PYTORCH_VERSION=0.4.0
RUN set -x && \
    pip install --no-cache-dir http://download.pytorch.org/whl/cu90/torch-${PYTORCH_VERSION}-cp36-cp36m-linux_x86_64.whl && \
    http_proxy=$PIP_PROXY pip install --no-cache-dir torchvision

# Keras+TensorFlow
# https://github.com/uber/horovod/blob/master/Dockerfile
ARG TENSORFLOW_VERSION=1.9.0
RUN http_proxy=$PIP_PROXY pip install --no-cache-dir tensorflow-gpu==$TENSORFLOW_VERSION
RUN http_proxy=$PIP_PROXY pip install --no-cache-dir keras==2.2.2

# horovod
RUN set -x && \
    ldconfig /usr/local/cuda/lib64/stubs && \
    http_proxy=$PIP_PROXY HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod && \
    ldconfig

# その他Pythonライブラリ色々
RUN set -x && \
    http_proxy=$PIP_PROXY pip install --no-cache-dir \
        'git+https://github.com/cocodataset/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI' \
        'git+https://www.github.com/keras-team/keras-contrib.git' \
        'scikit-optimize[plots]' \
        Augmentor \
        Flask \
        Flask-Migrate \
        Flask-SQLAlchemy \
        albumentations \
        autopep8 \
        better_exceptions \
        catboost \
        diskcache \
        fasteners \
        fire \
        flake8 \
        flake8-docstrings \
        flake8-pep257 \
        gunicorn \
        hyperopt \
        imageio \
        imbalanced-learn \
        janome \
        jupyterlab \
        kaggle \
        kaggle-cli \
        keras-rl \
        lightgbm \
        nltk \
        opencv-python \
        pandas-profiling \
        pip-tools \
        pycodestyle==2.3.1 \
        pylint \
        pytest \
        pytest-timeout \
        pytest-xdist \
        python-dotenv \
        tabulate \
        tqdm \
        tslearn \
        xgboost \
        && \
    jupyter serverextension enable --py jupyterlab --sys-prefix

# monkey patch
COPY sitecustomize.py /usr/local/lib/python3.6/

# ・sshd用ディレクトリ作成
# ・horovod用のNCCL / OpenMPI設定
# ・cuda、python、caffeなどのパスを通す
# ・matplotlibがエラーにならないようにMPLBACKEND=Aggを設定
# ・sudoでhttp_proxyなどが引き継がれるようにしておく
# ・最後にldconfigしておく
RUN set -x && \
    mkdir -pm 744 /var/run/sshd && \
    echo NCCL_DEBUG=INFO >> /etc/nccl.conf && \
    echo NCCL_SOCKET_IFNAME=^docker0 >> /etc/nccl.conf && \
    echo 'hwloc_base_binding_policy = none' >> /usr/local/etc/openmpi-mca-params.conf && \
    echo 'rmaps_base_mapping_policy = slot' >> /usr/local/etc/openmpi-mca-params.conf && \
    echo 'btl_tcp_if_exclude = lo,docker0' >> /usr/local/etc/openmpi-mca-params.conf && \
    echo 'export PATH=/opt/caffe/build/tools:/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH' > /etc/profile.d/docker.sh && \
    echo 'export CAFFE_ROOT=/opt/caffe' >> /etc/profile.d/docker.sh && \
    echo 'export PYTHONPATH=/opt/caffe/python' >> /etc/profile.d/docker.sh && \
    echo 'export MPLBACKEND=Agg' >> /etc/profile.d/docker.sh && \
    echo 'Defaults env_keep += "http_proxy https_proxy ftp_proxy no_proxy"' > /etc/sudoers.d/docker && \
    echo 'Defaults always_set_home' >> /etc/sudoers.d/docker && \
    chmod 0440 /etc/sudoers.d/* && \
    visudo --check && \
    ldconfig

# sshd以外の使い方をするとき用
ENV PATH="/opt/caffe/build/tools:$PATH" \
    CAFFE_ROOT="/opt/caffe" \
    PYTHONPATH="/opt/caffe/python" \
    MPLBACKEND="Agg"

COPY start_sshd.sh /root/
RUN date '+%Y/%m/%d %H:%M:%S' > /image.version
CMD ["/root/start_sshd.sh"]
