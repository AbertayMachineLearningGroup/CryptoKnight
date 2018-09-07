# Deep Learning Based Cryptographic Primitive Classification

Automated cryptographic classification framework using Intel's [Pin](https://software.intel.com/en-us/articles/pintool-downloads) platform for dynamic binary instrumentation and [PyTorch](http://pytorch.org/) for deep learning.

* Clone Repository
* Required Python libraries: ```sudo apt-get install python-pip, python-tk```
* Install requirements: ```pip install -r requirements.txt```
* Install toolkit: ```python knight.py --setup```
* Binary compilation requires [OpenSSL](https://www.openssl.org/): ```sudo apt install libssl-dev```

Automatically draw distribution:
```
python crypto.py -d scale
```

Evaluatation:
```
python knight.py --predict <executable>
python knight.py --evaluate <dataset>
```

To add custom cryptographic samples to the generation pool, please follow the [Format Specification](https://github.com/gregdhill/honours/blob/master/data/config/README.md).
