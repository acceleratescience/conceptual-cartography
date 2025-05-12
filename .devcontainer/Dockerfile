FROM mcr.microsoft.com/devcontainers/python:3.12

ENV POETRY_VIRTUALENVS_IN_PROJECT=true

# Install Starship prompt
RUN curl -sS https://starship.rs/install.sh | sh -s -- --yes \
    && echo 'eval "$(starship init bash)"' >> /home/vscode/.bashrc

# Create a minimal Starship config
RUN mkdir -p /home/vscode/.config \
    && echo '[directory]\ntruncation_length = 3\n\n[git_branch]\nformat = "[$branch]($style) "' > /home/vscode/.config/starship.toml