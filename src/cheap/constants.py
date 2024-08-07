import os
from pathlib import Path


# DEFAULT_CACHE = Path(os.environ['HOME']) / f".cache/cheap"
DEFAULT_CACHE = Path("/data/lux70/cheap")


# Weights to trained latent-to-sequence decoder
DECODER_CKPT_PATH = Path(DEFAULT_CACHE) / "sequence_decoder/mlp.ckpt"


# Directory to where per-channel statistics are stored
TENSOR_STATS_DIR = Path(DEFAULT_CACHE) / "statistics"


# Directory to where pre-trained models are stored
CHECKPOINT_DIR_PATH = Path(DEFAULT_CACHE) / "checkpoints"


# Mapping of compression levels to model IDs
CATH_COMPRESS_LEVEL_TO_ID = {
    2: {
        4: "8ebs7j9h",
        8: "mm9fe6x9",
        16: "kyytc8i9",
        32: "fbbrfqzk",
        64: "13lltqha",
        128: "uhg29zk4",
        256: "ich20c3q",
        512: "7str7fhl",
        1024: "g8e83omk",
    },
    1: {
        4: "1b64t79h",
        8: "1hr1x9r5",
        16: "yfel5fnl",
        32: "v2cer77t",
        64: "2tjrgcde",
        128: "3rs1hxky",
        256: "5z4iaak9",
        512: "q3m9fhii",
    },
}
