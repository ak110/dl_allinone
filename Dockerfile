FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04

# 実行時に残さないようにENVではなくARGでnoninteractive
ARG DEBIAN_FRONTEND=noninteractive
# ビルド中はエラーなどをググりやすいように英語
ENV LANG C.UTF-8

# command-not-found対策
RUN set -x && \
    rm -f /etc/apt/apt.conf.d/docker-gzip-indexes

# apt
RUN set -x && \
    sed -ie 's@http://archive.ubuntu.com/ubuntu/@http://ftp.riken.go.jp/Linux/ubuntu/@g' /etc/apt/sources.list && \
    sed -ie 's@^deb-src@# deb-src@g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install --yes --no-install-recommends \
        apt-transport-https \
        apt-utils \
        build-essential \
        ca-certificates \
        curl \
        locales \
        software-properties-common \
        wget \
        && \
    locale-gen ja_JP.UTF-8 && \
    update-locale LANG=ja_JP.UTF-8 LANGUAGE='ja_JP:ja' && \
    add-apt-repository ppa:git-core/ppa && \
    wget -q https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh -O- | bash && \
    apt-get install --yes --no-install-recommends git git-lfs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# https://github.com/tensorflow/addons#c-custom-op-compatibility-matrix
# > GCC 7.3.1
ARG GPP_VERSION=7
# aptその2
# python用: libbluetooth-dev, tk-dev, uuid-dev
# opencv用: libopencv-dev, libgtk2.0-dev
# scipyビルド用: gfortran
RUN set -x && \
    apt-get update && \
    apt-get install --yes --no-install-recommends \
        accountsservice \
        ack-grep \
        apparmor \
        apt-file \
        automake \
        bash-completion \
        bc \
        bind9-host \
        bsdmainutils \
        busybox-initramfs \
        busybox-static \
        cifs-utils \
        cmake \
        command-not-found \
        console-setup \
        console-setup-linux \
        corkscrew \
        cpio \
        cron \
        dbus \
        debconf-i18n \
        dialog \
        distro-info-data \
        dmidecode \
        dmsetup \
        dnsutils \
        dosfstools \
        ed \
        eject \
        emacs \
        entr \
        file \
        fonts-ipafont \
        fonts-liberation \
        friendly-recovery \
        ftp \
        fuse \
        g++-$GPP_VERSION \
        gdb \
        geoip-database \
        gettext-base \
        gfortran \
        graphviz \
        groff-base \
        hdf5-tools \
        hdparm \
        htop \
        iftop \
        imagemagick \
        inetutils-traceroute \
        info \
        init \
        initramfs-tools \
        initramfs-tools-bin \
        initramfs-tools-core \
        install-info \
        iotop \
        iproute2 \
        iptables \
        iputils-ping \
        iputils-tracepath \
        irqbalance \
        isc-dhcp-client \
        isc-dhcp-common \
        jq \
        kbd \
        keyboard-configuration \
        klibc-utils \
        kmod \
        krb5-locales \
        language-pack-ja \
        language-selector-common \
        less \
        libatlas-base-dev \
        libbluetooth-dev \
        libboost-dev \
        libboost-filesystem-dev \
        libboost-system-dev \
        libbz2-dev \
        libffi-dev \
        libgdbm-dev \
        libgtk2.0-dev \
        libjpeg-dev \
        liblzma-dev \
        libmecab-dev \
        libncurses5-dev \
        libopencv-dev \
        libpng-dev \
        libprotobuf-dev \
        libreadline-dev \
        libsqlite3-dev \
        libssl-dev \
        libtool \
        libwebp-dev \
        libyaml-dev \
        linux-base \
        logrotate \
        lshw \
        lsof \
        ltrace \
        man-db \
        manpages \
        mecab \
        mecab-ipadic-utf8 \
        mecab-jumandic-utf8 \
        mime-support \
        mlocate \
        mtr-tiny \
        nano \
        net-tools \
        netbase \
        netcat-openbsd \
        netplan.io \
        networkd-dispatcher \
        nplan \
        ntfs-3g \
        nvidia-opencl-dev \
        openssh-client \
        openssh-server \
        openssl \
        p7zip-full \
        pandoc \
        parted \
        pciutils \
        plymouth \
        plymouth-theme-ubuntu-text \
        popularity-contest \
        powermgmt-base \
        protobuf-compiler \
        psmisc \
        publicsuffix \
        rsync \
        rsyslog \
        screen \
        shared-mime-info \
        sl \
        smbclient \
        strace \
        subversion \
        sudo \
        swig \
        systemd \
        systemd-sysv \
        tcl-dev \
        tcpdump \
        telnet \
        tesseract-ocr \
        tesseract-ocr-jpn \
        tesseract-ocr-jpn-vert \
        tesseract-ocr-script-jpan \
        tesseract-ocr-script-jpan-vert \
        texlive-fonts-recommended \
        texlive-plain-generic \
        texlive-xetex \
        time \
        tk-dev \
        tmux \
        tmuxinator \
        tzdata \
        ubuntu-advantage-tools \
        ubuntu-minimal \
        ubuntu-release-upgrader-core \
        ubuntu-standard \
        ucf \
        udev \
        ufw \
        unzip \
        update-manager-core \
        usbutils \
        uuid-dev \
        uuid-runtime \
        valgrind \
        vim \
        whiptail \
        xauth \
        xdg-user-dirs \
        xkb-data \
        xxd \
        xz-utils \
        zip \
        zlib1g-dev \
        zsh \
        && \
    # MeCabの標準はIPA辞書にしておく
    update-alternatives --set mecab-dictionary /var/lib/mecab/dic/ipadic-utf8 && \
    # 後始末
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# MKL, IPP
# https://software.intel.com/en-us/articles/installing-intel-free-libs-and-python-apt-repo
RUN set -x && \
    wget -q https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB -O- | apt-key add - && \
    wget -q https://apt.repos.intel.com/setup/intelproducts.list -O /etc/apt/sources.list.d/intelproducts.list && \
    apt-get update && \
    apt-get install --yes intel-mkl-64bit-2020.0-088 intel-ipp-64bit-2020.0-088 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "/opt/intel/mkl/lib/intel64" >> /etc/ld.so.conf.d/intel.conf && \
    ldconfig && \
    echo ". /opt/intel/bin/compilervars.sh intel64" > /etc/profile.d/intel.sh

# OpenMPI
# https://github.com/horovod/horovod/blob/master/Dockerfile.gpu
# https://www.open-mpi.org/software/
RUN set -x && \
    mkdir /tmp/openmpi &&\
    cd /tmp/openmpi && \
    wget -q "https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.5.tar.bz2" -O openmpi.tar.bz2 && \
    echo "d85ad67fb1d5decd78a4bb883874e289 *openmpi.tar.bz2" | md5sum -c - && \
    tar xfj openmpi.tar.bz2 && \
    cd openmpi-4.0.5 && \
    ./configure --with-cuda --disable-mpi-fortran --disable-java --enable-orterun-prefix-by-default && \
    make -j$(nproc) all && \
    make -j$(nproc) install && \
    make -j$(nproc) distclean && \
    ldconfig && \
    rm -rf /tmp/openmpi

# devpi-server用
ARG PIP_TRUSTED_HOST=""
ARG PIP_INDEX_URL=""

# python
# https://github.com/docker-library/python/blob/master/3.8/buster/Dockerfile
ARG PYTHON_VERSION="3.8.7"
RUN set -ex && \
    wget -O python.tar.xz "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz" && \
    mkdir /tmp/python && \
    tar -xJC /tmp/python --strip-components=1 -f python.tar.xz && \
    rm python.tar.xz && \
    cd /tmp/python && \
    ./configure --build="$(dpkg-architecture --query DEB_BUILD_GNU_TYPE)" \
        --enable-loadable-sqlite-extensions \
        --enable-optimizations \
        --enable-option-checking=fatal \
        --enable-shared \
        --with-system-expat \
        --with-system-ffi \
        --with-ensurepip=upgrade \
        && \
    make -j "$(nproc)" && \
    make install && \
    ldconfig && \
    find /usr/local -depth \
        \( \
            \( -type d -a \( -name test -o -name tests -o -name idle_test \) \) \
            -o \( -type f -a \( -name '*.pyc' -o -name '*.pyo' -o -name '*.a' \) \) \
            -o \( -type f -a -name 'wininst-*.exe' \) \
        \) -exec rm -rf '{}' + && \
    rm -rf /tmp/python && \
    ln -s /usr/local/bin/idle3 /usr/local/bin/idle && \
    ln -s /usr/local/bin/pydoc3 /usr/local/bin/pydoc && \
    ln -s /usr/local/bin/python3 /usr/local/bin/python && \
    ln -s /usr/local/bin/python3 /usr/local/bin/python-config && \
    ln -s /usr/local/bin/pip3 /usr/local/bin/pip && \
    python3 --version
# pip
RUN set -ex && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir wheel

# numpy/scipy with MKL
# バージョン: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/pip_package/setup.py#L77
RUN set -x && \
    echo '[mkl]' > /root/.numpy-site.cfg && \
    echo 'library_dirs = /opt/intel/mkl/lib/intel64' >> /root/.numpy-site.cfg && \
    echo 'include_dirs = /opt/intel/mkl/include' >> /root/.numpy-site.cfg && \
    echo 'mkl_libs = mkl_rt' >> /root/.numpy-site.cfg && \
    echo 'lapack_libs =' >> /root/.numpy-site.cfg && \
    pip install --no-binary :all: numpy~=1.19.2 scipy~=1.5.2

# h5py<3.0.0: https://github.com/tensorflow/tensorflow/issues/44467
RUN set -x && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        numpy~=1.19.2 scipy~=1.5.2 \
        Augmentor \
        Flask \
        Flask-Login \
        Flask-Migrate \
        Flask-Restless \
        Flask-SQLAlchemy \
        Keras \
        Pillow \
        albumentations \
        autopep8 \
        bandit \
        bashplotlib \
        bcrypt \
        better_exceptions \
        black \
        bokeh \
        cairocffi \
        catboost \
        category_encoders \
        chainer \
        chainerrl \
        cookiecutter \
        cupy-cuda110 \
        cysignals \
        cython \
        diskcache \
        editdistance \
        efficientnet\>=1.1.0 \
        eli5 \
        ensemble-boxes \
        fasteners \
        fastprogress \
        fastrlock \
        feather-format \
        featuretools \
        fire \
        flake8 \
        flake8-bugbear \
        fugashi[unidic,unidic-lite] \
        gensim \
        gluoncv \
        gluonnlp \
        gunicorn \
        h5py\<3.0.0 \
        hyperopt \
        image-classifiers \
        imageio \
        imbalanced-learn \
        imgaug \
        imgdup \
        ipadic \
        ipywidgets \
        iterative-stratification \
        janome \
        japanize-matplotlib \
        jpholiday \
        jumandic \
        kaggle \
        keras2onnx \
        matplotlib \
        mecab-python3 \
        mpi4py \
        # CUDA 11.0対応待ち (horovodにも影響するため注意)
        # mxnet-cu110mkl \
        mypy \
        natsort \
        nlp \
        nltk \
        noise \
        nose \
        numba \
        onnxmltools \
        opencv-python \
        openpyxl \
        optuna\>=1.3.0 \
        pandas \
        pandas-profiling \
        passlib \
        pip-tools \
        pipdeptree \
        pipenv \
        plotly \
        poetry \
        pre-commit \
        progressbar2 \
        pycryptodome \
        # nlp
        pyarrow\>=0.16.0,\<1.0.0 \
        pydot \
        pygments \
        pylama \
        pylama_pylint \
        pylint \
        pyod \
        pypandoc \
        pytesseract \
        pytest \
        pytest-cookies \
        pytest-timeout \
        pytest-xdist \
        python-dotenv \
        python-utils \
        pyupgrade \
        pyyaml \
        recommonmark \
        requests \
        rgf_python \
        rope \
        safety \
        scikit-image \
        scikit-learn \
        scikit-optimize[plots] \
        seaborn \
        segmentation-models \
        sentencepiece \
        setup-cfg-fmt \
        signate \
        six \
        sklearn_pandas \
        spacy \
        sphinx \
        sphinx-autobuild \
        sphinx-autodoc-typehints \
        sphinx_rtd_theme \
        stickytape \
        sudachidict_core \
        sudachipy \
        sympy \
        tabulate \
        tensorflow-addons[tensorflow] \
        tensorflow-datasets \
        tensorflow-hub \
        tensorflow~=2.4.0 \
        tensorpack \
        # https://github.com/explosion/spaCy/issues/2883
        # https://github.com/explosion/spaCy/blob/master/requirements.txt
        thinc==7.4.1 \
        tf2cv \
        tf2onnx \
        tqdm \
        tsfresh \
        # mypy用バージョン指定。なぜかchainerのPython2用の依存関係に従ってしまう？
        typing-extensions\>=3.7.4 \
        xgboost \
        xlrd \
        xlwt \
        yapf \
        # 依存関係に注意
        chainercv

# PyTorch関連: https://pytorch.org/get-started/locally/
RUN set -x && \
    # PyTorchが既にインストールされてしまっていないことの確認
    test $(pip freeze | grep ^torch== | wc -l) -eq 0 && \
    # PyTorchとそれに依存するものたちのインストール
    pip install --no-cache-dir \
        allennlp \
        cnn-finetune \
        fastai \
        pretrainedmodels \
        pytorch-ignite \
        pytorch-lightning \
        tokenizers \
        torch==1.7.1+cu110 \
        torchaudio==0.7.2 \
        torchtext \
        torchvision==0.8.2+cu110 \
        transformers[ja] \
        --find-links=https://download.pytorch.org/whl/torch_stable.html

# apex
RUN set -x && \
    git clone --depth=1 https://github.com/NVIDIA/apex.git /tmp/apex &&\
    cd /tmp/apex &&\
    pip install --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ &&\
    rm -rf /tmp/apex

# 依存関係の問題があって後回しなやつ
RUN set -x && \
    pip install --no-cache-dir \
        fasttext \
        lycon \
        ptk \
        pycocotools \
        tslearn \
        ;

# 辞書など
RUN set -x && \
    python3 -m nltk.downloader -d /usr/local/share/nltk_data popular
RUN set -x && \
    python3 -m spacy download en --no-cache
RUN set -x && \
    python3 -m unidic download

# nodejs
ARG NODEJS_VERSION=v12.18.3
RUN set -x && \
    wget -q -O- https://nodejs.org/dist/$NODEJS_VERSION/node-$NODEJS_VERSION-linux-x64.tar.xz | tar xJ -C /tmp/ && \
    mv /tmp/node-$NODEJS_VERSION-linux-x64/bin/* /usr/local/bin/ && \
    mv /tmp/node-$NODEJS_VERSION-linux-x64/lib/* /usr/local/lib/ && \
    mv /tmp/node-$NODEJS_VERSION-linux-x64/include/* /usr/local/include/ && \
    mv /tmp/node-$NODEJS_VERSION-linux-x64/share/doc/* /usr/local/share/doc/ && \
    mv /tmp/node-$NODEJS_VERSION-linux-x64/share/man/man1/* /usr/local/share/man/man1 && \
    rm -rf /tmp/node-$NODEJS_VERSION-linux-x64

# jupyter関連
# plotly: https://plot.ly/python/getting-started/#jupyterlab-support-python-35
RUN set -x && \
    pip install --no-cache-dir \
        jupyter-tensorboard \
        jupyterlab~=2.0 \
        jupyterlab-git \
        && \
    export NODE_OPTIONS=--max-old-space-size=4096 &&\
    (jupyter labextension install --debug-log-path=/tmp/jupyterlab-build.log \
        @jupyter-widgets/jupyterlab-manager \
        @jupyterlab/git \
        @jupyterlab/toc \
        jupyterlab-chart-editor \
        jupyterlab-plotly \
        jupyterlab_tensorboard \
        plotlywidget \
        || (cat /tmp/jupyterlab-build.log && false)) && \
    jupyter serverextension enable --sys-prefix --py \
        jupyterlab \
        jupyterlab_git

# LightGBM
# 参考: https://github.com/microsoft/LightGBM/issues/586
RUN set -x && \
    mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd && \
    pip install --no-cache-dir --no-binary :all: --install-option=--gpu lightgbm

# horovod
# 参考: https://github.com/horovod/horovod/blob/master/Dockerfile.gpu
RUN set -x && \
    ldconfig /usr/local/cuda/lib64/stubs && \
    HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITH_MXNET=0 \
        pip install --no-cache-dir horovod && \
    ldconfig

# 最後にPillow-SIMD
RUN set -x && \
    CC="cc -mavx2" pip install --no-cache-dir --force-reinstall Pillow-SIMD

# npm
RUN set -x && \
    npm install -g pyright npm-check-updates

# ユーザー作成
ARG RUN_USER=user
ARG RUN_UID=1000
RUN set -x && \
    useradd --create-home --shell=/bin/bash --uid=$RUN_UID $RUN_USER

RUN set -x && \
    # sshd用ディレクトリ作成
    mkdir --mode=744 /var/run/sshd && \
    # sshdで~/.ssh/environmentがあれば読み込むようにする
    sed -i 's/#PermitUserEnvironment no/PermitUserEnvironment yes/' /etc/ssh/sshd_config && \
    # horovod用のNCCL設定
    echo 'NCCL_DEBUG=INFO' >> /etc/nccl.conf && \
    # horovod用のOpenMPI設定
    echo 'hwloc_base_binding_policy = none' >> /usr/local/etc/openmpi-mca-params.conf && \
    echo 'rmaps_base_mapping_policy = slot' >> /usr/local/etc/openmpi-mca-params.conf && \
    echo 'btl_tcp_if_exclude = lo,docker0' >> /usr/local/etc/openmpi-mca-params.conf && \
    # 環境変数設定
    echo 'export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH' > /etc/profile.d/docker.sh && \
    echo 'export BETTER_EXCEPTIONS=1' >> /etc/profile.d/docker.sh && \
    echo 'export TF_FORCE_GPU_ALLOW_GROWTH=true' >> /etc/profile.d/docker.sh && \
    # sudoでhttp_proxyなどが引き継がれるようにしておく
    echo 'Defaults env_keep += "http_proxy https_proxy ftp_proxy no_proxy PIP_TRUSTED_HOST PIP_INDEX_URL"' > /etc/sudoers.d/docker && \
    echo 'Defaults always_set_home' >> /etc/sudoers.d/docker && \
    # $RUN_USERをパスワード無しでsudoできるようにしておく
    echo $RUN_USER ALL=\(ALL\) NOPASSWD:ALL >> /etc/sudoers.d/docker　&&\
    chmod 0440 /etc/sudoers.d/* && \
    visudo --check && \
    # completion
    poetry completions bash > /etc/bash_completion.d/poetry.bash-completion && \
    # 最後にldconfigしておく
    ldconfig

# sshd以外の使い方をするとき用環境変数色々
ENV TZ='Asia/Tokyo' \
    LANG='ja_JP.UTF-8' \
    PYTHONIOENCODING='utf-8' \
    PYTHONDONTWRITEBYTECODE=1 \
    BETTER_EXCEPTIONS=1 \
    TF_FORCE_GPU_ALLOW_GROWTH='true'

# SSHホストキーを固定で用意
COPY --chown=root:root .ssh_host_keys/ssh_host_* /etc/ssh/
# 作った日時を記録しておく (一応)
RUN date '+%Y/%m/%d %H:%M:%S' > /image.version
# sshd
# -D: デタッチしない
# -e: ログを標準エラーへ
CMD ["/usr/sbin/sshd", "-D", "-e"]
