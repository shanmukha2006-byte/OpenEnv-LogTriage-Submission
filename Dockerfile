# Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy entire project into the container
COPY . .

# Upgrade pip and install the project (non-editable)
RUN pip install --upgrade pip
RUN pip install .

# Expose the FastAPI server port
EXPOSE 7860

# Run the 'server' console script defined in pyproject.toml
CMD ["server"]
