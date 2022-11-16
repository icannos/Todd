# Todd: Text Out of Distribution Detection

Todd is a library designed to provide anomaly scorer and the derived filters for text
generation. It supports three levels of anomaly detection: input level, generated sequence 
level and token level. The input level anomaly detection is used for OOD detection. Whereas 
the generated sequence level aims at selecting the best candidates in a beam and the token
level aims at detecting potentially wrong tokens in the generated sequence.


## Installation

### From github

```bash
git clone git@github.com:icannos/Todd.git
cd Todd
pip install -e .
```

### From pip

```bash
pip install todd
```


## API