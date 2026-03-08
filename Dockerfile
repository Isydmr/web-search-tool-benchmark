FROM continuumio/miniconda3:latest

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gfortran \
    libopenblas-dev \
    liblapack-dev && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Create conda environment
RUN conda create -n myenv python=3.10 && \
    conda run -n myenv pip install --upgrade pip && \
    conda run -n myenv conda install -y -c conda-forge \
        numpy \
        pandas \
        scikit-learn \
        transformers \
        spacy && \
    conda run -n myenv conda install pytorch=2.1.0 torchvision torchaudio -c pytorch && \
    conda run -n myenv pip install \
        fastapi==0.104.1 \
        uvicorn==0.24.0 \
        sqlalchemy==2.0.23 \
        python-dotenv==1.0.0 \
        together==0.2.4 \
        schedule==1.2.1 \
        openai==1.70.0 \
        deepeval>=2.5.2 \
        sentence-transformers==4.0.2 \
        strsimpy==0.2.1 \
        python-multipart==0.0.6 \
        jinja2==3.1.2 \
        aiohttp==3.9.1 \
        feedparser>=6.0.0 \
        beautifulsoup4>=4.9.1 \
        dateparser>=1.1.0 \
        requests>=2.31.0 \
        openai==1.70.0 \
        transformers>=4.31.0

# Install pygooglenews dependencies
RUN conda run -n myenv pip install beautifulsoup4>=4.9.1 dateparser>=0.7.6 requests>=2.28.1 feedparser>=6.0.8
RUN conda run -n myenv pip install accelerate

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p app/static app/templates

# Make run.sh executable
RUN chmod +x run.sh

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
# Set default command to use conda environment
SHELL ["/bin/bash", "-c"]
CMD source activate myenv && ./run.sh
