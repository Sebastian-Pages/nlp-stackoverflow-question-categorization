# Use the official Python image as a base
FROM python:3.10
RUN pip install nltk
RUN python -m nltk.downloader all
# FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file first for better caching of dependencies
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet averaged_perceptron_tagger

# RUN pip install nltk
# RUN [ "python", "-c", "import nltk; nltk.download('all')" ]
# RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"

# Install NLTK and download necessary resources to a specific directory
# RUN pip install nltk
# RUN python -c "import nltk; \
#                 nltk.download('stopwords', download_dir='/root/nltk_data'); \
#                 nltk.download('punkt', download_dir='/root/nltk_data'); \
#                 nltk.download('wordnet', download_dir='/root/nltk_data'); \
#                 nltk.download('averaged_perceptron_tagger', download_dir='/root/nltk_data')"

# RUN pip install nltk && \
#     mkdir ~/nltk_data && \
#     mkdir ~/nltk_data/chunkers && \
#     mkdir ~/nltk_data/corpora && \
#     mkdir ~/nltk_data/taggers && \
#     mkdir ~/nltk_data/tokenizers && \
#     python -c "import nltk; nltk.download(['punkt', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words'])"


# Set NLTK data directory environment variable
ENV NLTK_DATA /root/nltk_data

# Copy the rest of your application code to the container
COPY . .

# Expose the port that the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]