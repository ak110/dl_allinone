#!/bin/bash
#
# 環境変数:
# - SSH_USER
# - SSH_UID
# - SSH_KEY or SSH_PASS (どちらか必須)
#
set -ux

if [ ! -e /root/.docker-initialized -a -v SSH_USER ] ; then
    # ユーザ作成
    useradd --create-home --shell=/bin/bash --uid=$SSH_UID $SSH_USER
    cp -rn /etc/skel/. /home/$SSH_USER
    if [ -v SSH_KEY ] ; then
        mkdir -pm 700 /home/$SSH_USER/.ssh
        if [ -e /home/$SSH_USER/.ssh/authorized_keys ] ; then
            SSH_KEY="$(echo "$SSH_KEY" | sed 's@ssh-rsa \([a-zA-Z0-9/+=]*\) .*@ssh-rsa \1@')"
            grep "$SSH_KEY" /home/$SSH_USER/.ssh/authorized_keys > /dev/null
            if [ $? != 0 ] ; then
                echo "$SSH_KEY" >> /home/$SSH_USER/.ssh/authorized_keys
            fi
        else
            echo "$SSH_KEY" >> /home/$SSH_USER/.ssh/authorized_keys
        fi
        chown -R $SSH_USER:users /home/$SSH_USER/.ssh
        chmod 600 /home/$SSH_USER/.ssh/authorized_keys
    else
        echo "$SSH_USER:$SSH_PASS" | chpasswd
    fi
    chown $SSH_USER:users /home/$SSH_USER /home/$SSH_USER/.ssh

    # sudo
    echo $SSH_USER ALL=\(ALL\) NOPASSWD:ALL > /etc/sudoers.d/$SSH_USER

    # オーナー設定
    if [ -d /data ] ; then
        chown $SSH_USER:users /data
    fi

    # ライブラリパス更新
    ldconfig

    touch /root/.docker-initialized
fi

# sshd
# -D: デタッチしない
# -e: ログを標準エラーへ
exec /usr/sbin/sshd -D -e
