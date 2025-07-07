FROM ubuntu:noble

USER root

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        git zsh curl wget ca-certificates \
    && apt-get clean \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*
RUN yes | unminimize

ARG HOME=/root
ARG ZSH=${HOME}/.oh-my-zsh
ARG ZSH_CUSTOM=${ZSH}/custom
WORKDIR ${HOME}


RUN <<EOT
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
    mkdir -p ${ZSH_CUSTOM}
    git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM}/plugins/zsh-autosuggestions
    git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM}/plugins/zsh-syntax-highlighting
    sed -i.bak 's/plugins=(\(.*\))/plugins=(\1 zsh-autosuggestions zsh-syntax-highlighting)/' ~/.zshrc
EOT


ARG SPACESHIP_REMOTE=https://github.com/spaceship-prompt/spaceship-prompt.git
RUN <<EOT
    git clone ${SPACESHIP_REMOTE} ${ZSH_CUSTOM}/themes/spaceship-prompt --depth=1
    ln -s ${ZSH_CUSTOM}/themes/spaceship-prompt/spaceship.zsh-theme ${ZSH_CUSTOM}/themes/spaceship.zsh-theme
    sed -i 's|ZSH_THEME="robbyrussell"|ZSH_THEME="spaceship"|g' .zshrc
EOT


RUN <<EOT
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo 'eval "$(uv generate-shell-completion zsh)"' >> ~/.zshrc
    echo 'eval "$(uvx --generate-shell-completion zsh)"' >> ~/.zshrc
EOT

# Install Bazel for DeepMind Control Suite.
RUN <<EOT
    wget -q https://github.com/bazelbuild/bazelisk/releases/download/v1.26.0/bazelisk-linux-amd64 -O /usr/local/bin/bazel
    chmod +x /usr/local/bin/bazel
    bazel
EOT
