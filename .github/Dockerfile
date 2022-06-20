FROM python:3.9-slim as setup
RUN apt update && \
    apt install -y r-base && \
    rm -rf /var/lib/apt/lists/*

RUN python3.9 -m pip install --no-cache-dir --upgrade pip

FROM setup as install

COPY requirements.txt .
RUN python3.9 -m pip install --no-cache-dir -r requirements.txt

COPY dist dist
RUN python3.9 -m pip install --no-cache-dir dist/*.whl

FROM setup
RUN R -e 'install.packages(c("isa2", "biclust"), repos="https://cloud.r-project.org/")'

WORKDIR /biclust_sandbox
COPY example.py .
COPY --from=install /usr/local/lib/python3.9/site-packages/ /usr/local/lib/python3.9/site-packages/

ENTRYPOINT ["/bin/bash"]
