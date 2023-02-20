mkdir ./data
mkdir ./data/datasets
mkdir ./data/experiments
cd ./data/datasets/
wget https://dataset.ait.ethz.ch/downloads/IMavatar_data/data/yufeng.zip
unzip yufeng.zip
cd ../experiments/
wget https://dataset.ait.ethz.ch/downloads/IMavatar_data/checkpoint/yufeng.zip
unzip yufeng.zip
## download the other subject with the following commands
# cd ../datasets/
# wget https://dataset.ait.ethz.ch/downloads/IMavatar_data/data/marcel.zip
# unzip marcel.zip
# cd ../experiments/
# wget https://dataset.ait.ethz.ch/downloads/IMavatar_data/checkpoint/IMavatar_marcel.zip
# unzip IMavatar_marcel.zip
