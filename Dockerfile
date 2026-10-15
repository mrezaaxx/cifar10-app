FROM python:3.11-slim

# Hugging Face Spaces menjalankan container sebagai user UID 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    TF_CPP_MIN_LOG_LEVEL=2
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 7860
# XSRF dimatikan karena Space disajikan lewat iframe/proxy HF;
# tanpa ini st.file_uploader gagal dengan error 403.
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
