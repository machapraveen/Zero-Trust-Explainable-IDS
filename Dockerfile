# ZT-XAIIDS: Explainable Deep Learning Framework for Zero-Trust Intrusion Detection
# Department of CSE - AIML | Aditya College of Engineering and Technology

FROM python:3.10-slim

LABEL project="Explainable Deep Learning Framework for Zero-Trust Intrusion Detection Using Multi-Level Network Traffic Analytics"
LABEL institution="Aditya College of Engineering and Technology - Department of CSE (AIML)"
LABEL version="2.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    MPLBACKEND=Agg \
    FLASK_ENV=production \
    FLASK_APP=wsgi:application \
    PORT=8000 \
    WORKERS=2

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -r ztxai

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=ztxai:ztxai . .

RUN mkdir -p /app/instance /app/logs /app/data/processed /app/models \
    && chown -R ztxai:ztxai /app

USER ztxai

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers ${WORKERS} --threads 4 --timeout 120 --keep-alive 5 --log-level info --access-logfile - --error-logfile - wsgi:application"]
