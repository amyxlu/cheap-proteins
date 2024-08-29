# CHEAP (Compressed Hourglass Embedding Adaptation of Proteins)

Code for [Tokenized and Continuous Embedding Compressions of Protein Sequence and Structure](https://www.biorxiv.org/content/10.1101/2024.08.06.606920v1).

![Overview of the CHEAP model.](cheap.png)


## Demo
For a demo of reported results, including phenomena of massive activations in [ESMFold (Lin et al.)](https://www.science.org/doi/10.1126/science.ade2574), see `notebooks/cheap_example.ipynb`.

Code for [Tokenized and Continuous Embedding Compressions of Protein Sequence and Structure](https://www.biorxiv.org/content/10.1101/2024.08.06.606920v1).

![Overview of the CHEAP model.](cheap.png)


## Demo
For a demo of reported results, including phenomena of massive activations in [ESMFold (Lin et al.)](https://www.science.org/doi/10.1126/science.ade2574), see `notebooks/cheap_example.ipynb`.

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
# fork contains minor changes to setup.py to use C++17 instead of C++14
# for newer versions of PyTorch
git clone https://github.com/amyxlu/openfold.git
# fork contains minor changes to setup.py to use C++17 instead of C++14
# for newer versions of PyTorch
git clone https://github.com/amyxlu/openfold.git
cd openfold
python setup.py develop
```

## Usage

To obtain compressed representations of sequences (also see notebook example at `notebooks/usage_example.ipynb`):

```
import torch
device = torch.device("cuda")

# replace with shorten factor and dimension of choice
from cheap.pretrained import CHEAP_shorten_1_dim_64
pipeline = CHEAP_shorten_1_dim_64(return_pipeline=True)

# sample sequences
# note: returned representation will be padded to the length of the longest sequence
# consider cropping the sequences beforehand if memory is an issue.

sequences = [
    # >cath|current|12asA00/4-330
    "AYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAV",
    # >cath|current|132lA00/2-129
    "VFGRCELAAAMRHGLDNYRGYSLGNWVCAAFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKIVSDGNGMNAWVAWRNRCGTDVQAWIRGCRL",
    # >cath|current|153lA00/1-185
    "RTDCYGNVNRIDTTGASCKTAKPEGLSYCGVSASKKIAERDLQAMDRYKTIIKKVGEKLCVEPAVIAGIISRESHAGKVLKNGWGDRGNGFGLMQVDKRSHKPQGTWNGEVHITQGTTILINFIKTIQKKFPSWTKDQQLKGGISAYNAGAGNVRSYARMDIGTTHDDYANDVVARAQYYKQHGY",
]

emb, mask = pipeline(sequences)
```



~Upon public release of model weights, the weights will be automatically downloaded to `~/.cache/cheap`.
For now, the compression model weights, decoder model weights, and per-channel statistics for normalization must be manually placed in `~/.cache/cheap`.
Alternatively, one can update the `DEFAULT_CACHE` variable in `constants.py` to where the weights are stored.
Public release of model weights is pending legal approval.
Internal users can access relevant weights at `/data/lux70/data/cheap`.~

**Sept 2024:** Weights are available here: https://huggingface.co/amyxlu/cheap-proteins/tree/main. Will get to making the weights easier to load in the next weeks, but in the meantime, the dedicated user can hack their way around this.

## Citation

If this code is useful in your work, please use the citation:

```
@article{lu2024tokenized,
  title={Tokenized and Continuous Embedding Compressions of Protein Sequence and Structure},
  author={Lu, Amy X and Yan, Wilson and Yang, Kevin K and Gligorijevic, Vladimir and Cho, Kyunghyun and Abbeel, Pieter and Bonneau, Richard and Frey, Nathan},
  journal={bioRxiv},
  pages={2024--08},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```

Contact: amyxlu [at] berkeley [dot] edu OR lux70 [at] gene [dot] com.
