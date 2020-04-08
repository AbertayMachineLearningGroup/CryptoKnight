# Deep Learning Based Cryptographic Primitive Classification

Automated cryptographic classification framework using Intel's [Pin](https://software.intel.com/en-us/articles/pintool-downloads) platform for dynamic binary instrumentation and [PyTorch](http://pytorch.org/) for deep learning.

* Clone Repository
* Required Python libraries: ```sudo apt-get install python-pip python-tk```
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

To add custom cryptographic samples to the generation pool, please follow the [Format Specification](data/config/README.md).

We also published "CryptoKnight: Generating and Modelling Compiled Cryptographic Primitives
" that can be found [here](http://www.mdpi.com/2078-2489/9/9/231) in Open Access.

If you want to cite the paper please use the following format;

````
@Article{info9090231,
AUTHOR = {Hill, Gregory and Bellekens, Xavier},
TITLE = {CryptoKnight: Generating and Modelling Compiled Cryptographic Primitives},
JOURNAL = {Information},
VOLUME = {9},
YEAR = {2018},
NUMBER = {9},
ARTICLE NUMBER = {231},
URL = {http://www.mdpi.com/2078-2489/9/9/231},
ISSN = {2078-2489},
DOI = {10.3390/info9090231}
}
````
