# Backend image. Base must be able to run the OMR stack + LilyPond.
# Build: docker build -t jianpu-api .
FROM condaforge/miniforge3:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
      lilypond git && rm -rf /var/lib/apt/lists/*

# SumisoraOMR repo (OMR + jianpu-ly pipeline). Pin to your fork/commit.
RUN git clone https://github.com/YOUR_USER/SumisoraOMR.git /app/sumisora
WORKDIR /app/sumisora
RUN conda env create -f environment.yml || pip install -r requirements.txt

COPY requirements.txt /app/api-requirements.txt
RUN conda run -n sumisora pip install -r /app/api-requirements.txt

ENV SUMISORA_ROOT=/app/sumisora

COPY app.py /app/app.py
EXPOSE 8000
CMD ["conda", "run", "--no-capture-output", "-n", "sumisora", \
     "uvicorn", "--app-dir", "/app", "app:app", "--host", "0.0.0.0", "--port", "8000"]
