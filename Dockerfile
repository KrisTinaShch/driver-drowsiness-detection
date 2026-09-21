# Service image: the eye-state classifier behind an HTTP endpoint.
#
# Build (from the project root, with best.pt present):
#   docker build -t eye-state .
# Run:
#   docker run --rm -p 8000:8000 eye-state
# Then open http://localhost:8000/docs

FROM python:3.11-slim

WORKDIR /app

# CPU-only torch. The wheel on PyPI pulls the CUDA runtime with it, which is
# dead weight here: there is no GPU inside this container.
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Copied separately from the source: this layer is cached and only rebuilt
# when the dependency list changes, not on every edit of service.py.
COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt

# --no-cache-dir above: pip would otherwise keep a copy of every downloaded
# wheel inside the image. Useful on a laptop, pure waste in an image.

# Do not run as root: a process that only answers HTTP has no need for it.
# The user is created before the files are copied, so that COPY can set the
# owner directly.
RUN useradd --create-home app

# --chown instead of a later `chown -R`: layers only ever add, they cannot
# change a file in place. Rewriting the owner afterwards would store the whole
# 43 MB of weights a second time.
COPY --chown=app:app service.py best.pt ./

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
