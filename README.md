# Deep Supervised Discrete Hashing

## REQUIREMENTS
`pip install -r requirements.txt`

1. pytorch 1.1
2. loguru

## DATASETS
1. [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
2. [NUS-WIDE](https://pan.baidu.com/s/1f9mKXE2T8XpIq8p7y8Fa6Q) Password: uhr3

## USAGE
```
usage: run.py [-h] [--dataset DATASET] [--root ROOT] [--batch-size BATCH_SIZE]
              [--arch ARCH] [--lr LR] [--code-length CODE_LENGTH]
              [--max-iter MAX_ITER] [--num-query NUM_QUERY]
              [--num-train NUM_TRAIN] [--num-workers NUM_WORKERS]
              [--topk TOPK] [--gpu GPU] [--mu MU] [--nu NU] [--eta ETA]
              [--evaluate-interval EVALUATE_INTERVAL]

DSDH_PyTorch

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Dataset name.
  --root ROOT           Path of dataset
  --batch-size BATCH_SIZE
                        Batch size.(default: 128)
  --arch ARCH           CNN model name.(default: alexnet)
  --lr LR               Learning rate.(default: 1e-5)
  --code-length CODE_LENGTH
                        Binary hash code length.(default: 12,24,32,48)
  --max-iter MAX_ITER   Number of iterations.(default: 150)
  --num-query NUM_QUERY
                        Number of query data points.(default: 1000)
  --num-train NUM_TRAIN
                        Number of training data points.(default: 5000)
  --num-workers NUM_WORKERS
                        Number of loading data threads.(default: 6)
  --topk TOPK           Calculate map of top k.(default: all)
  --gpu GPU             Using gpu.(default: False)
  --mu MU               Hyper-parameter.(default: 1e-2)
  --nu NU               Hyper-parameter.(default: 1)
  --eta ETA             Hyper-parameter.(default: 1e-2)
  --evaluate-interval EVALUATE_INTERVAL
                        Evaluation interval.(default: 10)
```

# Experiments
CNN model: Alexnet. Compute mean average precision(MAP).

cifar10: 1000 query images, 5000 training images.

nus-wide-tc21: 21 classes, 2100 query images, 10500 training images.

 bits | 12 | 24 | 32 | 48  
   :-:   |  :-:    |   :-:   |   :-:   |   :-:     
cifar10@ALL | 0.6994 | 0.7204 | 0.7267 | 0.7345
nus-wide@5000 | 0.8052| 0.8317 | 0.8332 | 0.8437
