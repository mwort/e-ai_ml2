#!/usr/bin/env python
# coding: utf-8

# # 01[Product Owner] Create a new "project" (== Gitlab repository)
# ![Gitlab screenshot](slides/01a.png "Create a new repository") or 
# ![Gitlab screenshot](slides/01b.png "Create a new repository")

# # 02 [Product Owner] Create blank project
# ![Gitlab screenshot](slides/02.png)

# # 03 [Product Owner] Set project name and create project
# ![Gitlab screenshot](slides/03.png)

# # 04 [DevOps Engineer] Configure project
# ![Gitlab screenshot](slides/04.png)
# ## Go to section "Visibility, project features, permissons"
# ![Gitlab screenshot](slides/05.png)
# ## Enable CI and container registry
# ![Gitlab screenshot](slides/06_1.png)
# ![Gitlab screenshot](slides/06_2.png)
# ## Save changes
# ![Gitlab screenshot](slides/06_3.png)

# # 05 [DevOps Engineer] Settings -> Configure CI/CD
# ![Gitlab screenshot](slides/07.png)
# ## Go to section "Runners"
# ![Gitlab screenshot](slides/08_1.png)
# ## Enable Instance runners provided by DKRZ
# ![Gitlab screenshot](slides/08_2.png)
# (This setting is saved automatically)

# # 06 [Data Scientist/ML Engineer] Commit your code
# ## get Gitlab SSH URL
# ![Gitlab screenshot](slides/09.png)
# Register your public ssh key in your Gitlab profile settings

# # [Data Scientist/ML Engineer] Add README.md + first code version

# In[1]:


get_ipython().system('git clone git@gitlab.dkrz.de:b380572/face-detection-for-python-demo.git')


# In[2]:


get_ipython().run_line_magic('cd', 'face-detection-for-python-demo')


# In[3]:


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

# In[5]:


get_ipython().system('git status')


# In[6]:


get_ipython().system('git add README.md face.py requirements.txt')
get_ipython().system('git commit -m "I have used the entirety of my skillset to develop this face recognition demo code."')


# In[7]:


get_ipython().system('git push')


# # [DevOps Engineer] Implement Black code formatter
# [https://black.readthedocs.io/en/stable/](https://black.readthedocs.io/en/stable/)

# In[8]:


get_ipython().system('pip install black')
get_ipython().system('echo black >> requirements.txt')


# In[9]:


get_ipython().system('black face.py')


# In[10]:


get_ipython().system('cat face.py')


# In[11]:


get_ipython().system('git diff')


# In[12]:


get_ipython().system('git add face.py requirements.txt')
get_ipython().system('git commit -m "Apply Black"')


# # [DevOps Engineer] Implement Black CI test and Commit

# In[13]:


# Define CI in Gitlab
with open(".gitlab-ci.yml", "w") as f:
    f.write(r'''
stages:
  - lint
  - build # for later use
  - test # for later use

lint:
  stage: lint
  tags:
    - sphinx, dkrz
  script:
    - export
    - pip install black
    - black --check . # returns 1, if code is unformatted
''')


# In[14]:


get_ipython().system('git add .gitlab-ci.yml')
get_ipython().system('git commit -m "Add Black CI"')
get_ipython().system('git push')


# # Anyone can verify the code's conformance. [e.g. Product Owners and Analysts]
# ![Gitlab screenshot](slides/10.png)

# # Anyone can verify the code's conformance. [e.g. Product Owners and Analysts]
# ![Gitlab screenshot](slides/11.png)

# # Anyone can verify the code's conformance. [e.g. Product Owners and Analysts]
# ![Gitlab screenshot](slides/12.png)

# # [Software Engineer] Suggests CLI (comand line interface)
# So far the application only processes Apollo_11_Crew.jpg

# In[15]:


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


# In[16]:


get_ipython().system('git checkout -b CLI')
get_ipython().system('git diff')


# In[17]:


get_ipython().system('echo "typer" >> requirements.txt')
get_ipython().system('git add face.py requirements.txt')
get_ipython().system('git commit -m "Implement simple CLI"')
get_ipython().system('git push --set-upstream origin CLI')


# # [Software Engineer] Create Merge Request for CLI
# ![Gitlab screenshot](slides/13.png)

# ![Gitlab screenshot](slides/14.png)

# ![Gitlab screenshot](slides/15.png)

# ![Gitlab screenshot](slides/16.png)

# ![Gitlab screenshot](slides/17.png)

# ![Gitlab screenshot](slides/18.png)

# # [DevOps Engineer] Reviews MR
# The DevOps Engineer knows black by heart and knows how to satisfy black.
# ![Gitlab screenshot](slides/19.png)
# ![Gitlab screenshot](slides/20.png)
# ![Gitlab screenshot](slides/21.png)
# ![Gitlab screenshot](slides/22.png)

# # [Software Engineer] Fixes style
# ![Gitlab screenshot](slides/23.png)
# ![Gitlab screenshot](slides/24.png)

# ![Gitlab screenshot](slides/25.png)
# ![Gitlab screenshot](slides/26.png)

# # e.g. Product Owner merges MR
# ![Gitlab screenshot](slides/27.png)

# In[18]:


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

# In[19]:


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

CMD face.py
''')


# # [Software Engineer] Build Docker container with Kaniko in Gitlab CI

# In[20]:


with open(".gitlab-ci.yml", "a") as f:
    f.write(r'''
variables:
  CONTAINER_TAG: "${CI_COMMIT_SHA}"

build-docker:
  stage: build
  tags:
    - docker-any-image, dkrz
  image:
    name: gcr.io/kaniko-project/executor:debug
    entrypoint: [""]
  script:
    - echo "{\"auths\":{\"$CI_REGISTRY\":{\"username\":\"$CI_REGISTRY_USER\",\"password\":\"$CI_REGISTRY_PASSWORD\"}}}" > /kaniko/.docker/config.json
    - /kaniko/executor --reproducible --context $CI_PROJECT_DIR
        --dockerfile $CI_PROJECT_DIR/Dockerfile
        --destination $CI_REGISTRY_IMAGE:$CONTAINER_TAG
  #only: # restrict to main branch
  #  - main
''')


# # [Software Engineer] Convert Docker image to SIF (Singularity image format)

# In[21]:


with open(".gitlab-ci.yml", "a") as f:
    f.write(r'''
export-sif:
  stage: build
  needs: ["build-docker"]
  tags:
    - docker-any-image, dkrz
  image:
    name: singularityware/singularity
    entrypoint: [""]
  script:
    - SINGULARITY_DOCKER_USERNAME=$CI_REGISTRY_USER
        SINGULARITY_DOCKER_PASSWORD=$CI_REGISTRY_PASSWORD
        singularity pull docker://$CI_REGISTRY_IMAGE:$CONTAINER_TAG
    - apk add curl
    - ls -l
    - 'curl --header "JOB-TOKEN: $CI_JOB_TOKEN" --upload-file ${CI_PROJECT_NAME}_${CONTAINER_TAG}.sif ${CI_API_V4_URL}/projects/${CI_PROJECT_ID}/packages/generic/images/0.0.1/${CI_PROJECT_NAME}_${CONTAINER_TAG}.sif'
  #only: # restrict to main branch
  #  - main
''')


# In[22]:


get_ipython().system('git add Dockerfile .gitlab-ci.yml')
get_ipython().system('git commit -m "Build containers in CI"')
get_ipython().system('git push')


# ![Gitlab screenshot](slides/28.png)
# ![Gitlab screenshot](slides/29.png)

# ![Gitlab screenshot](slides/31.png)
# ![Gitlab screenshot](slides/32.png)

# # Package Registry: SIF for apptainer
# ![Gitlab screenshot](slides/30.png)

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




