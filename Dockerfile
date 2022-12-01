FROM continuumio/anaconda3

# Add environment.yml to the build context and create the environment
ARG conda_env=NLP
ADD requirements.txt /tmp/requirements.txt
RUN conda create --name ${conda_env} python==3.10.6

# Activating the environment and starting the jupyter notebook
RUN echo "source activate ${conda_env}" > ~/.bashrc
ENV PATH /opt/conda/envs/${conda_env}/bin:$PATH

RUN pip install -r /tmp/requirements.txt

# Start jupyter server on container
EXPOSE 8888
ENTRYPOINT ["jupyter","notebook","--ip=0.0.0.0","--port=8888","--allow-root","--no-browser"]