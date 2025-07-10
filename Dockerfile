FROM python:3.9.18

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    gnupg \
    unzip \
    xvfb \
    libxi6 \
    libgconf-2-4 \
    libnss3 \
    libxss1 \
    libasound2 \
    fonts-noto-cjk \
    fonts-noto-cjk-extra \
    fonts-ipafont-gothic \
    fonts-wqy-zenhei \
    fonts-thai-tlwg \
    fonts-kacst \
    fonts-freefont-ttf && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt and install python libraries
COPY requirements.txt .

RUN pip install -r requirements.txt

# Copy the project
COPY . .

# Terminal command
CMD [ "python", "main.py" ]  # FIXME: the directory of the main.py file