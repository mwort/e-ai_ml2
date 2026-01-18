# 02_code_to_execute_in_docker.py

from pathlib import Path

print("Hello from Docker")

# Write a file into the mounted directory
out = Path("02_docker_output.txt")
out.write_text("This file was generated inside a Docker container.\n")

print(f"Wrote file: {out.resolve()}")
