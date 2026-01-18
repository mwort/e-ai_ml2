#!/usr/bin/env python
# coding: utf-8

# # Introduction

# ![Introduction](slides/i01.png)

# In[1]:


#!wget https://upload.wikimedia.org/wikipedia/commons/3/3d/Apollo_11_Crew.jpg # by NASA
from fdlite import FaceDetection, FaceDetectionModel
from fdlite.render import Colors, detections_to_render_data, render_to_image
import PIL

def detect_faces(image: PIL.Image):
    detect_faces = FaceDetection(model_type=FaceDetectionModel.BACK_CAMERA)
    faces=detect_faces(image)
    print(f"Found {len(faces)} faces")
    return faces

def mark_faces(image_filename):
    """Mark all faces recognized in the image"""
    image=PIL.Image.open(image_filename)

    faces=detect_faces(image)

    # Draw faces
    render_data = detections_to_render_data(faces,bounds_color=Colors.GREEN,line_width=3)
    render_to_image(     render_data, image)

    image.save(image_filename + '.out.jpg')

mark_faces("Apollo_11_Crew.jpg")


# ![Introduction](Apollo_11_Crew.jpg)

# ![Introduction](Apollo_11_Crew_out.jpg)

# # 01[Product Owner] Create a new "project" (== GitHub repository)
# ![Gitlab screenshot](slides/01GH.png "Create a new repository")

# # (02 [Product Owner] Create blank project)
# Gitlab only

# # 03 [Product Owner] Set project name and create project
# ![Gitlab screenshot](slides/03GH.png)
# 
# Note: Choose `public` to enable Package registry (used for container).

# # (04 [DevOps Engineer] Initialize repository)
# ![Gitlab screenshot](slides/04GH.png)

# In[2]:


get_ipython().system('mkdir face-detection-for-python-demo')
get_ipython().run_line_magic('cd', 'face-detection-for-python-demo')
get_ipython().system('echo "# Face-Detection-For-Python-Demo" >> README.md')
get_ipython().system('git init')
get_ipython().system('git add README.md')
get_ipython().system('git commit -m "first commit"')
get_ipython().system('git branch -M main')
get_ipython().system('git remote add origin git@github.com:MeraX/Face-Detection-For-Python-Demo.git')
get_ipython().system('git push -u origin main')


# # (05 [DevOps Engineer] Settings -> Configure CI/CD)
# GitLab only

# # 06 [Data Scientist/ML Engineer] Commit your code

# # [Data Scientist/ML Engineer] Add README.md + first code version

# In[ ]:


#!git clone git@github.com:MeraX/Face-Detection-For-Python-Demo.git
#%cd face-detection-for-python-demo


# In[8]:


# create ML application
with open("face.py", "w") as f:
    f.write(r'''
from fdlite import FaceDetection, FaceDetectionModel
from fdlite.render import Colors, detections_to_render_data, render_to_image
import PIL

def detect_faces(image: PIL.Image):
    detect_faces = FaceDetection(model_type=FaceDetectionModel.BACK_CAMERA)
    faces=detect_faces(image)
    print(f"Found {len(faces)} faces")
    return faces

def mark_faces(image_filename):
    """Mark all faces recognized in the image"""
    image=PIL.Image.open(image_filename)

    faces=detect_faces(image)

    # Draw faces
    render_data = detections_to_render_data(faces,bounds_color=Colors.GREEN,line_width=3)
    render_to_image(     render_data, image)

    image.save(image_filename.with_suffix('.out.jpg'))

mark_faces("Apollo_11_Crew.jpg")
    ''')


# In[4]:


# copy prepared README
get_ipython().system('cp ../README.md .')

# note requirements
get_ipython().system(' echo "git+https://github.com/seppe-intelliprove/face-detection-onnx" > requirements.txt')


# # [Data Scientist/ML Engineer] Commit README.md + first code version

# In[9]:


get_ipython().system('git status')


# In[10]:


get_ipython().system('git add README.md face.py requirements.txt')
get_ipython().system('git commit -m "I have used the entirety of my skillset to develop this face recognition demo code."')


# In[11]:


get_ipython().system('git push')


# # [DevOps Engineer] Implement Black code formatter
# [https://black.readthedocs.io/en/stable/](https://black.readthedocs.io/en/stable/)

# In[12]:


get_ipython().system('pip install black')
get_ipython().system('echo black >> requirements.txt')


# In[13]:


get_ipython().system('black face.py')


# In[14]:


get_ipython().system('cat face.py')


# In[15]:


get_ipython().system('git diff')


# In[16]:


get_ipython().system('git add face.py requirements.txt')
get_ipython().system('git commit -m "Apply Black"')
get_ipython().system('git push')


# # [DevOps Engineer] Implement Black CI test and Commit

# In[17]:


get_ipython().run_line_magic('mkdir', '-p .github/workflows/')
# Define CI in Gitlab
with open(".github/workflows/pytest.yml", "w") as f:
    f.write(r'''
name: lint

on: [push]

jobs:
  lint-job:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10"]
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install black
    - name: Analysing the code with black
      run: |
        black --check . # returns 1, if code is unformatted
''')


# In[18]:


get_ipython().system('git add .github/workflows/pytest.yml')
get_ipython().system('git commit -m "Add Black CI"')
get_ipython().system('git push')


# # Anyone can verify the code's conformance. [e.g. Product Owners and Analysts]
# ![Gitlab screenshot](slides/10GH.png)

# ![Gitlab screenshot](slides/11GH.png)

# ![Gitlab screenshot](slides/12GH.png)

# # [Software Engineer] Suggests CLI (comand line interface)
# So far the application only processes Apollo_11_Crew.jpg

# In[20]:


# create ML application
with open("face.py", "w") as f:
    f.write(r'''#!/usr/bin/env python3
from fdlite import FaceDetection, FaceDetectionModel
from fdlite.render import Colors, detections_to_render_data, render_to_image

import PIL
import typer
import pathlib


def detect_faces(image: PIL.Image):
    detect_faces = FaceDetection(model_type=FaceDetectionModel.BACK_CAMERA)
    faces = detect_faces(image)
    print(f"Found {len(faces)} faces")
    return faces


def mark_faces(image_filename: pathlib.Path):
    """Mark all faces recognized in the image"""
    image = PIL.Image.open(image_filename)

    faces = detect_faces(image)

    # Draw faces
    render_data = detections_to_render_data(
        faces, bounds_color=Colors.GREEN, line_width=3
    )
    render_to_image(render_data, image)

    image.save(image_filename.with_suffix(".out.jpg"))


if __name__ == '__main__': typer.run( mark_faces )
''')


# In[21]:


get_ipython().system('git checkout -b CLI')
get_ipython().system('git diff')


# In[22]:


get_ipython().system('echo "typer" >> requirements.txt')
get_ipython().system('git add face.py requirements.txt')
get_ipython().system('git commit -m "Implement simple CLI"')
get_ipython().system('git push --set-upstream origin CLI')


# # [Software Engineer] Create Pull Request for CLI
# ![Gitlab screenshot](slides/14GH.png)

# ![Gitlab screenshot](slides/15GH.png)

# ![Gitlab screenshot](slides/17GH.png)

# ![Gitlab screenshot](slides/18GH.png)

# # [DevOps Engineer] Reviews MR
# The DevOps Engineer knows black by heart and knows how to satisfy black.
# ![Gitlab screenshot](slides/19GH.png)
# ![Gitlab screenshot](slides/20GH.png)
# ![Gitlab screenshot](slides/21GH.png)
# ![Gitlab screenshot](slides/22GH.png)
# 
# <!--
# ```suggestion
# if __name__ == '__main__':
#     typer.run(mark_faces)
# ```
# DevOps Engineer: This should satisfy black.
# -->

# # [Software Engineer] Fixes style
# ![Gitlab screenshot](slides/23GH.png)

# ![Gitlab screenshot](slides/25GH.png)
# ![Gitlab screenshot](slides/26GH.png)

# # e.g. Product Owner merges MR
# ![Gitlab screenshot](slides/27aGH.png)
# ![Gitlab screenshot](slides/27bGH.png)

# In[23]:


get_ipython().system('git checkout main')
get_ipython().system('git pull')


# # Towards operational use
# * Face recognition should be easy to deploy
# * Portable
# * Isolated environments with controlled dependencies
# * Self contained
# * Version control of binary code for easy rollback
# 
# ## --> Glorified solution: Container
# * Mainstream technology: Docker container
# * Docker requires local `sudo` privileges to build and run containers
# * Building: we use Kaniko to build Docker container image via Gitlab CI instead
#   * Available at DKRZ and gitlab.dwd.de
# * Running: using Apptainer (formally Singularity)
#   * Available on HPC and Workstation (MetSD ticket)

# # [Software Engineer] Container description: Dockerfille

# In[24]:


with open("Dockerfile", "w") as f:
    f.write(r'''
FROM python:3.10-slim-bookworm
ARG DEBIAN_FRONTEND=noninteractive

USER root
# Run basesetup
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash dwd
USER dwd

WORKDIR /home/dwd

# Set environment variables for the virtual environment
ENV VIRTUAL_ENV=/home/dwd/venv
# Make virtual environment accessible
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN cd /home/dwd && python3 -m venv $VIRTUAL_ENV

COPY --chown=dwd requirements.txt .
RUN pip3 install -r requirements.txt

# Add face.py, make it executable and add it to PATH.
COPY --chown=dwd face.py .
RUN chmod 775 face.py && ln -s $PWD/face.py $VIRTUAL_ENV/bin/face.py

ENTRYPOINT ["face.py"]
''')


# # [Software Engineer] Build Docker container with Kaniko in Gitlab CI

# In[25]:


with open(".github/workflows/docker-image.yml", "w") as f:
    f.write(r'''
name: Docker Image CI

on:
  push:
    branches: [ "main" ]
  # Do the builds on all pull requests (to test them)
  pull_request: []

permissions:
  packages: write

jobs:
  build:
    runs-on: ubuntu-latest
    env:
      GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    steps:
    - uses: actions/checkout@v4
    - name: Build the Docker image
      run: |
        echo "Using $(docker -v)"

        echo "${GITHUB_TOKEN}" | docker login ghcr.io -u "${{ github.actor }}" --password-stdin

        export IMAGE_NAME=$(echo ${GITHUB_REPOSITORY} | tr '[:upper:]' '[:lower:]')
        docker build . --file Dockerfile --tag ghcr.io/${IMAGE_NAME}:${GITHUB_SHA}
        docker push ghcr.io/${IMAGE_NAME}:${GITHUB_SHA}
''')


# # [Software Engineer] Convert Docker image to SIF (Singularity image format)

# In[26]:


with open(".github/workflows/export-sif.yml", "w") as f:
    f.write(r'''
name: Export sif

on:
  workflow_run:
    workflows: ["Docker Image CI"]
    branches: [main]
    types: 
      - completed

permissions:
  packages: write

jobs:
  to-sif:
    runs-on: ubuntu-latest
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    container:
      image: quay.io/singularity/singularity:v3.8.3
    strategy:
      fail-fast: false
    env:
      GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    steps:
    - uses: actions/checkout@v4
    - name: Build the Docker image
      run: |
        echo "Using $(singularity version)"

        echo "${GITHUB_TOKEN}" | singularity remote login -u ${{ github.actor }} --password-stdin oras://ghcr.io

        export IMAGE_NAME=$(echo ${GITHUB_REPOSITORY} | tr '[:upper:]' '[:lower:]')

        singularity pull docker://ghcr.io/${IMAGE_NAME}:${GITHUB_SHA}
        singularity push *.sif oras://ghcr.io/${IMAGE_NAME}:sif_${GITHUB_SHA}
''')


# In[27]:


get_ipython().system('git add Dockerfile .github/workflows/docker-image.yml .github/workflows/export-sif.yml')
get_ipython().system('git commit -m "Build containers in CI"')
get_ipython().system('git push')


# ![Gitlab screenshot](slides/28GH.png)
# ![Gitlab screenshot](slides/29GH.png)

# ![Gitlab screenshot](slides/31GH.png)
# ![Gitlab screenshot](slides/32GH.png)

# # Package Registry: get SIF for apptainer
# ![Gitlab screenshot](slides/33GH.png)
# Use the presented command, but use `apptainer` instead of `docker`!
# ```bash
# # Retrieve sif from Github Registry 
# apptainer pull ghcr.io/merax/face-detection-for-python-demo:sif_36c46c5066e04b1b208ed9beb8acdfb6d6bbc050
# ```

# # Download SIF and execute Apptainer
# 1. Manually download SIF with your browser.
# 2. `apptainer` options:
#     * `--cleanenv --env MALLOC_ARENA_MAX=2`: Do not import env of host-shell, technical stuff
#     * `--contain`: Minimal mounts
#     * `--no-home`: Do not mount `~/` (also included in `--contain`)
#     * `-B "$PWD:/mnt:rw"`: bind Option. Make host-PWD available as `/mnt` insinde container
#     * `--writable-tmpfs`: temporary writeable file system
#     * `--net --network none`: limit network access

# In[ ]:


# get test data:
get_ipython().system('cd .. && wget https://upload.wikimedia.org/wikipedia/commons/3/3d/Apollo_11_Crew.jpg # by NASA')
# run container
get_ipython().system('cd .. && apptainer exec      --cleanenv --env MALLOC_ARENA_MAX=2      --contain      -B "$PWD:/mnt:rw"      --writable-tmpfs      --net --network none      face-detection-for-python-demo_575fff2fa55fbe57467421205129cfbd84ff224f.sif face.py /mnt/Apollo_11_Crew.jpg')


# Result:
# ![Gitlab screenshot](./Apollo_11_Crew.out.jpg)

# In[ ]:




