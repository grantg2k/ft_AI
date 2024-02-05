FROM freqtradeorg/freqtrade:stable_freqaitorch
RUN pip install numpy technical datetime typing
RUN pip install -r requirements-hyperopt.txt

FROM freqtradeorg/freqtrade:develop_plot

# Install necessary packages including Java 17
USER root
RUN apt-get update && \
    apt-get install -y openjdk-17-jdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    
USER ftuser

# Pin jupyter-client to avoid tornado version conflict
RUN pip install jupyterlab jupyter-client==7.3.4 --user --no-cache-dir
RUN pip install -r requirements-dev.txt
#Additional Req

# Empty the ENTRYPOINT to allow all commands
ENTRYPOINT []
