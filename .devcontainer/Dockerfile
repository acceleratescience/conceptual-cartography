FROM mcr.microsoft.com/devcontainers/python:1-3.12

# Set working directory and copy project files
WORKDIR /workspace
COPY . .

RUN chmod +x ./setup.sh

# Run your universal setup script
RUN bash ./setup.sh --dev
