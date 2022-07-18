## requirement packages

# Dependencies:  <br />
python==3.7 <br />
torch==1.9.0+cu102 <br />
torchvision==0.10.0+cu102 <br />
torch-scatter==2.0.8 <br />
torch-sparse==0.6.11 <br />
torch-geometric==1.7.2 <br />
transformers==4.2.1 <br />
sckikit-learn==0.21.3 <br />
tqdm==4.62.0 <br />
numpy==1.19.5 <br />
pandas <br />
matplotlib==2.2.3 <br />
networkx==2.2 <br />
scipy==1.2.0 <br />
pyro-ppl==0.3.0 <br />
networkx <br />
pickle <br />
<br />

# required packages are in requirements.txt

```bash
pip install -r requirements.txt
```

# running environment


## data

All datasets are public accessible

[Twitter15](https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0 ) <br />
[Twitter16] (https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0 ) <br />
[CoAID] (https://github.com/cuilimeng/CoAID) version 0.4 <br />
[WEIBO] (https://alt.qcri.org/~wgao/data/rumdect.zip) <br />

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
This is the source code for 
[DUCK: Rumour Detection on Social Media by Modelling User and Comment Propagation Networks](https://aclanthology.org/2022.naacl-main.364/)


If you find this code useful, please let us know and cite our paper.  
If you have any question, please contact Lin at: s3795533 at student dot rmit dot edu dot au.
