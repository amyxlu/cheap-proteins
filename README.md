# CHEAP (Compressed Hourglass Embedding Adaptation of Proteins)

## Installation

Clone the repository:

```
git clone https://github.com/amyxlu/cheap-proteins.git
cd cheap-proteins
```

To create the environment for the repository:
```
conda env create --file environment.yaml
pip install -e .
```

**Important**: we use the frozen ESMFold structure module, which in turn uses the OpenFold implementation, and includes custom CUDA kernels for the attention mechanism. To trigger the build without the other OpenFold dependencies related to MSA construction, etc.:

```
git clone https://github.com/aqlaboratory/openfold.git
cd openfold
python setup.py develop
```

Alternatively, you can install OpenFold with:

```
pip install -q git+https://github.com/asarigun/openfold.git
```

(PRs to improve this hacky workaround is welcomed!)


### Using the Weights
Upon public release of model weights, the weights will be automatically downloaded to `~/.cache/cheap`.
For now, the compression model weights, decoder model weights, and per-channel statistics for normalization must be manually placed in `~/.cache/cheap`.
Alternatively, one can update the `DEFAULT_CACHE` variable in `constants.py` to where the weights are stored.
Public release of model weights is pending legal approval.
Internal users can access relevant weights at `/data/lux70/data/cheap`.

# Contact
amyxlu [at] berkeley [dot] edu OR lux70 [at] gene [dot] com
