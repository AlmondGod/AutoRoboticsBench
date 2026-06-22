FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN python -m pip install --upgrade pip && \
    python -m pip install pyyaml numpy

COPY docker/entrypoint.sh /usr/local/bin/robo_entrypoint.sh
RUN chmod +x /usr/local/bin/robo_entrypoint.sh

ENTRYPOINT ["/usr/local/bin/robo_entrypoint.sh"]
CMD ["sleep", "infinity"]
