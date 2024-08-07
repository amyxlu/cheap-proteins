# import time
# import re
# from pathlib import Path

# import wandb
# import hydra
# from omegaconf import DictConfig, OmegaConf
# import torch

# from cheap.esmfold import esmfold_v1
# from cheap.datasets import FastaDataModule, H5DataModule
# from cheap.utils import count_parameters
# from cheap.model import HourglassProteinCompressionTransformer
# import os


# shard_dir = "/data/lux70/data/cath/shards"
# dm = H5DataModule(shard_dir=shard_dir, embedder="esmfold", seq_len=128, batch_size=32, num_workers=0, dtype="fp32", shuffle_val_dataset=False)
# dm.setup()


# device = torch.device("cuda:0")
# model = HourglassProteinCompressionTransformer(dim=1024)

# train_dataloader = dm.train_dataloader()
# batch = next(iter(train_dataloader))


# # Helpers for loading latest checkpoint


# def find_latest_checkpoint(folder):
#     checkpoint_files = [f for f in os.listdir(folder) if f.endswith(".ckpt")]
#     if "last.ckpt" in checkpoint_files:
#         return "last.ckpt"

#     if not checkpoint_files:
#         return None

#     latest_checkpoint = max(checkpoint_files, key=lambda x: extract_step(x))
#     return latest_checkpoint


# def extract_step(checkpoint_file):
#     match = re.search(r"(\d+)-(\d+)\.ckpt", checkpoint_file)
#     if match:
#         return int(match.group(2))
#     return -1


# @hydra.main(version_base=None, config_path="configs", config_name="train_hourglass_vq")
# def train(cfg: DictConfig):
#     # general set up
#     torch.set_float32_matmul_precision("medium")

#     # maybe use prior job id, else generate new ID
#     if cfg.resume_from_model_id is not None:
#         job_id = cfg.resume_from_model_id
#         IS_RESUMED = True
#     else:
#         job_id = wandb.util.generate_id()
#         IS_RESUMED = False

#     # set up checkpoint and config yaml paths
#     dirpath = Path(cfg.paths.checkpoint_dir) / "hourglass_vq" / job_id
#     config_path = dirpath / "config.yaml"
#     if config_path.exists():
#         cfg = OmegaConf.load(config_path)
#         print("*" * 10, "\n", "Overriding config from job ID", job_id, "\n", "*" * 10)
#     else:
#         dirpath.mkdir(parents=True)
#         if not config_path.exists():
#             OmegaConf.save(cfg, config_path)

#     log_cfg = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)
#     print(OmegaConf.to_yaml(log_cfg))

#     # Data modules
#     start = time.time()
#     datamodule = hydra.utils.instantiate(cfg.datamodule)
#     datamodule.setup()
#     end = time.time()
#     print(f"Datamodule set up in {end - start:.2f} seconds.")

#     # maybe load esmfold if we need to normalize by channel
#     esmfold = None
#     if isinstance(datamodule, FastaDataModule):
#         esmfold = esmfold_v1()

#     # set up lightning module
#     model = hydra.utils.instantiate(cfg.hourglass, esmfold=esmfold)

#     trainable_parameters = count_parameters(model, require_grad_only=True)
#     log_cfg["trainable_params_millions"] = trainable_parameters / 1_000_000

#     if not cfg.dryrun:
#         # this will automatically log to the same wandb page
#         logger = hydra.utils.instantiate(cfg.logger, id=job_id)
#         # logger.watch(model, log="all", log_graph=False)
#     else:
#         logger = None

#     checkpoint_callback = hydra.utils.instantiate(cfg.callbacks.checkpoint, dirpath=dirpath)
#     lr_monitor = hydra.utils.instantiate(cfg.callbacks.lr_monitor)

#     callbacks = [checkpoint_callback, lr_monitor]

#     if cfg.use_compression_callback:
#         callbacks += [hydra.utils.instantiate(cfg.callbacks.compression, esmfold=esmfold)]  # creates ESMFold on CPU

#     trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

#     from plaid.utils import print_cuda_info
#     print_cuda_info()

#     if rank_zero_only.rank == 0 and isinstance(trainer.logger, WandbLogger):
#         trainer.logger.experiment.config.update({"cfg": log_cfg}, allow_val_change=True)

#     if not cfg.dryrun:
#         if IS_RESUMED:
#             # job id / dirpath was already updated to match the to-be-resumed directory
#             ckpt_fname = dirpath / find_latest_checkpoint(dirpath)
#             print("Resuming from ", ckpt_fname)
#             assert ckpt_fname.exists()
#             trainer.fit(model, datamodule=datamodule, ckpt_path=dirpath / ckpt_fname)
#         else:
#             trainer.fit(model, datamodule=datamodule)


# if __name__ == "__main__":
#     from plaid.utils import print_cuda_info
#     train()
