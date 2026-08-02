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

# 6. Install packages that task YAML files declare in `allowed_imports` but
#    that aren't in the Python standard library (e.g. circle_packing.yaml and
#    quantum_gravity_scaling.yaml both allow "numpy"; gravity_entanglement_
#    wick_derivation.yaml allows "sympy" -- without this, any evolved code
#    that imports them fails immediately inside the sandbox).
RUN pip install --no-cache-dir numpy sympy

# The default command can be omitted if the command is always specified during `docker run`
# For example, `python temp_script.py` will be the command passed by the EvaluatorAgent.
# CMD ["python"]
