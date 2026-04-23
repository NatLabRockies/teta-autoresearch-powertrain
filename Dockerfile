FROM ghcr.io/prefix-dev/pixi:latest

# Install git and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs curl ca-certificates && rm -rf /var/lib/apt/lists/* && \
    git lfs install

# Optional: inject corporate CA certificate for environments with SSL-intercepting proxies
ARG CA_CERT
RUN if [ -n "$CA_CERT" ]; then \
    echo "$CA_CERT" > /usr/local/share/ca-certificates/corporate-ca.crt && \
    update-ca-certificates; \
    fi
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install Claude Code
RUN npm install -g @anthropic-ai/claude-code

# Create a non-root user for sandboxing
RUN useradd --create-home --shell /bin/bash researcher

RUN git config --global user.email "researcher@example.com" && git config --global user.name "Auto Researcher"

# Clone the repo (requires a GitHub personal access token for github.nrel.gov)
ARG GIT_TOKEN
RUN git clone https://${GIT_TOKEN}@github.nrel.gov/RouteE/routee-autoresearch.git /workspace && \
    cd /workspace && git pull && git lfs pull

# Hand ownership to the non-root user before installing the environment
RUN chown -R researcher:researcher /workspace

USER researcher

# Set working directory
WORKDIR /workspace

RUN pixi install --frozen

# Default command: drop into a shell ready to run experiments
CMD ["bash"]
