
FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y \
        curl \
        wget \
        vim \
        nano \
        python3 \
        python3-pip \
        sudo && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt || true

RUN curl -fsSL https://ollama.com/install.sh | sh

CMD ["bash"]
