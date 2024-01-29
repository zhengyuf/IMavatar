mkdir ./data
mkdir ./data/datasets
mkdir ./data/experiments
cd ./data/datasets/
wget https://dataset.ait.ethz.ch/downloads/IMavatar_data/data/subject1.zip
unzip subject1.zip
cd ../experiments/
wget https://dataset.ait.ethz.ch/downloads/IMavatar_data/checkpoint/subject1.zip
unzip subject1.zip
## download the other subject with the following commands
# cd ../datasets/
# wget https://dataset.ait.ethz.ch/downloads/IMavatar_data/data/subject2.zip
# unzip subject2.zip
# cd ../experiments/
# wget https://dataset.ait.ethz.ch/downloads/IMavatar_data/checkpoint/subject2.zip
# unzip subject2.zip
