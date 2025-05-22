# 1. Start from a suitable Python base image
FROM python:3.10-slim

# 2. Set a working directory
WORKDIR /app

# 3. Copy the temp_script.py (which will be dynamically generated and mounted)
#    into this working directory.
#    While the actual script is mounted at runtime, WORKDIR ensures /app exists.
#    If specific files from the build context were needed in /app, they'd be copied here.
#    Example: COPY ./my_utility_script.py /app/my_utility_script.py

# 4. Ensure Python's output is unbuffered
ENV PYTHONUNBUFFERED=1

# 5. (Instruction for build process: name this image code-evaluator:latest)
#    docker build -t code-evaluator:latest .

# 6. For now, no additional libraries need to be installed.
#    If you need to install Python packages, uncomment and modify the following lines:
#    COPY requirements.txt .
#    RUN pip install --no-cache-dir -r requirements.txt
#
#    Make sure 'requirements.txt' is in the same directory as this Dockerfile (the build context).
#    Example requirements.txt content:
#    # numpy==1.23.5
#    # pandas==1.5.3

# The default command can be omitted if the command is always specified during `docker run`
# For example, `python temp_script.py` will be the command passed by the EvaluatorAgent.
# CMD ["python"]
