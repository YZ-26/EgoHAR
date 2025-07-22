#!/bin/bash
set -e
cd "$(dirname "$0")"

# Install Miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash

# Create & activate the Conda environment
~/miniconda3/bin/conda create -n egtea-env python==3.11 -y
source ~/miniconda3/etc/profile.d/conda.sh
conda activate egtea-env
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Install core requirements
pip install -r requirements.txt

conda install ipykernel --update-deps --force-reinstall -y

# Install EGTEA Gaze Plus Downloader
cd /workspace
mkdir -p data
git clone https://github.com/amitsou/EGTEA_Gaze_Plus_Downloader.git
cd EGTEA_Gaze_Plus_Downloader

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run the Scrapy spider
cd src/scrapper
scrapy crawl egtea_spider

# Go back to the root and process the data
cd ..
python main.py --trimmed_actions --out /workspace/data

rm -rf /workspace/EGTEA_Gaze_Plus_Downloader

# Preprocess the data
cd /workspace/EgoHAR
python preprocess.py

# Move files to the correct location
mv /workspace/EgoHAR/Gaze_Data /workspace/
mv /workspace/EgoHAR/train_split1_filtered.txt /workspace/
mv /workspace/EgoHAR/test_split1_filtered.txt /workspace/





