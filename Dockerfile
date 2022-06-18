FROM python:3.9 as setup
RUN apt update && \
    apt install -y r-base && \
    rm -rf /var/lib/apt/lists/*

FROM setup as build

RUN apt update && \
    apt install -y git python3-dev

RUN python3.9 -m pip install --no-cache-dir --upgrade pip && \
    python3.9 -m pip install --no-cache-dir build

COPY requirements.txt .
RUN python3.9 -m pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python3.9 -m build && \
    python3.9 -m pip install --no-cache-dir dist/*.whl

FROM setup
RUN R -e 'install.packages(c("isa2", "biclust"), repos="https://cloud.r-project.org/")'

WORKDIR /biclust_sandbox
COPY example.py .
COPY --from=build /usr/local/lib/python3.9/site-packages/ /usr/local/lib/python3.9/site-packages/
COPY --from=build /root/.cache/pip/ /root/.cache/pip/

ENTRYPOINT ["/bin/bash"]
