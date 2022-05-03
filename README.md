## requirement packages

# Dependencies:  
python==3.7
torch==1.9.0+cu102
torchvision==0.10.0+cu102
torch-scatter==2.0.8
torch-sparse==0.6.11
torch-geometric==1.7.2
transformers==4.2.1
sckikit-learn==0.21.3
tqdm==4.62.0
numpy==1.19.5
pandas
matplotlib==2.2.3
networkx==2.2
scipy==1.2.0
pyro-ppl==0.3.0
networkx
pickle


> required packages are in requirements.txt

```bash
pip install -r requirements.txt
```

# running environment


## data

All datasets are public accessible

[Twitter15](https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0 )
[Twitter16] (https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0 )
[CoAID] (https://github.com/cuilimeng/CoAID) version 0.4
[WEIBO] (https://alt.qcri.org/~wgao/data/rumdect.zip)

# data crawling tool

[twarc] (https://github.com/DocNow/twarc)


## training the DUCK model

```bash
python3 train.py --datasetName 'Twitter15' --baseDirectory './data' --mode 'DUCK' --modelName 'DUCK'
```

# comment graph data

```bash
python3 train.py --datasetName 'Twitter15' --baseDirectory './data' --mode 'CommentTree' --modelName 'Simple_GAT_BERT'
```

# user graph data

```bash
python3 train.py --datasetName 'Twitter15' --baseDirectory './data' --mode 'UserTree' --modelName 'Simple_GAT_BERT'
```

# run script

```
$ sh run.sh
```

## publicaton
This is the source code for DUCK: Rumour Detection on Social Media by Modelling User and Comment Propagation Networks
Paper accepted by NAACL 2022


If you find this code useful, please let us know and cite our paper.  
If you have any question, please contact Lin at: s3795533 at student dot rmit dot rmit dot edu dot au.
