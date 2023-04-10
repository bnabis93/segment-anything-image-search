PYTHON=3.9
BASENAME=$(shell basename $(CURDIR))
CONDA_CH=conda-forge defaults

env:
	conda create -n $(BASENAME) -y python=$(PYTHON) $(addprefix -c ,$(CONDA_CH))

setup:
	pip install -r requirements.txt

load-model:
	mkdir -p models
	curl -O https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth && mv sam_vit_h_4b8939.pth models/sam_vit_h_4b8939.pth