# syntax=docker/dockerfile:1

# Use a lightweight Python base image
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim-buster

# Prevent Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Prevent Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Create a non-privileged user to run the application
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the dependencies without cache
RUN pip install --no-cache-dir -r requirements.txt

# Switch to the non-privileged user
USER appuser

# Copy the rest of the application code to the container
COPY . .

# Expose the port that the application will run on
EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "__main__.py", "--server.port=8000"]
