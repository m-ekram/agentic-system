# Runs the eval suite in a clean, fixed environment — the same one CI uses.
# Proof that the project isn't quietly depending on something only installed on
# my laptop.
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first so Docker can cache the pip layer: editing a .py file
# then doesn't force a full reinstall on every rebuild.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default to the evals, which need no API key and no network.
# Run the agent instead with:
#   docker run --rm -it --env-file .env notes-agent python agent.py
CMD ["pytest", "-v"]
