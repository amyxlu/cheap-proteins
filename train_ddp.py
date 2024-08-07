# from cheap.datasets import FastaDataModule
# from cheap.model import HourglassProteinCompressionTransformer


# import time
# import re
# from pathlib import Path

# import wandb
# import hydra
# from omegaconf import DictConfig, OmegaConf
# import torch

# from cheap.esmfold import esmfold_v1
# from cheap.datasets import FastaDataModule
# from cheap.utils import count_parameters
# from cheap.typed import PathLike
# import os

# import torch.distributed as dist
# import torch.nn as nn
# import torch.optim as optim
# import torch.multiprocessing as mp

# from torch.nn.parallel import DistributedDataParallel as DDP


# # DDP helpers

# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12355'

#     # initialize the process group
#     dist.init_process_group("nccl", rank=rank, world_size=world_size)


# def cleanup():
#     dist.destroy_process_group()


# def train(rank: int, world_size: int, cfg: DictConfig, datamodule: FastaDataModule, checkpoint_dir: PathLike):
#     setup(rank, world_size)

#     # set up datamodule and distributed sampler
#     train_sampler = torch.utils.data.distributed.DistributedSampler(
#         datamodule.train_dataset, num_replicas=world_size, rank=rank
#     )
#     val_sampler = torch.utils.data.distributed.DistributedSampler(datamodule.val_dataset, shuffle=False, drop_last=True)
#     train_dataloader = datamodule.train_dataloader(sampler=train_sampler)
#     val_dataloader = datamodule.val_dataloader(sampler=val_sampler)

#     # set up model
#     model = HourglassProteinCompressionTransformer()

#     # set up DDP
#     model = DDP(model, device_ids=[rank])

#     # set up esmfold
#     esmfold = esmfold_v1()
#     esmfold = esmfold.to(rank)
#     esmfold = esmfold.eval().requires_grad_(False)

#     # set up optimizer
#     optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

#     # load checkpoint to rank, if applicable
#     latest_checkpoint = find_latest_checkpoint(checkpoint_dir)

#     if latest_checkpoint:
#         dist.barrier()
#         map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
#         checkpoint = torch.load(checkpoint_dir / latest_checkpoint, map_location=map_location)

#         best_loss = checkpoint["best_loss"]
#         model.load_state_dict(checkpoint['state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer'])
#         # scheduler.load_state_dict(checkpoint['scheduler'])
#         step = extract_step(latest_checkpoint)
#     else:
#         step = 0

#     # training loop
#     for epoch in range(cfg.num_epochs):
#         train_sampler.set_epoch(epoch)

#         for batch in train_dataloader:
#             optimizer.zero_grad()
#             loss = model(batch)
#             loss.backward()
#             optimizer.step()

#             if (step % cfg.log_interval == 0) and (rank == 0):
#                 wandb.log({"loss": loss.item()}, step=step)

#             # run eval every few steps
#             if (step % cfg.eval_interval == 0):
#                 with torch.no_grad():
#                     model.eval()
#                     validate()

#                 if rank == 0:
#                     wandb.log()

#                     # save checkpoint if best validation loss
#                     if (step % cfg.save_interval == 0):
#                         torch.save(
#                             {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
#                             f"epoch{epoch}-step{step}.ckpt",
#                         )

#             step += 1

#     cleanup()


# def validate(val_dataloader, world_size):

#     def run_validate(val_dataloader, ):
#         pass

#     # handle uneven batch sizes
#     if (len(val_dataloader.sampler) * world_size < len(val_dataloader.dataset)):
#         aux_val_dataset = torch.utils.data.Subset(
#             val_dataloader.dataset,
#             range(len(val_dataloader.sampler) * world_size, len(val_dataloader.dataset)))
#         val_dataloader.dataset = aux_val_dataset
#         run_validate(val_dataloader, len(val_dataloader))


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


# def setup_job_id(cfg: DictConfig):
#     # maybe use prior job id, else generate new ID
#     if cfg.resume_from_model_id is not None:
#         job_id = cfg.resume_from_model_id
#     else:
#         job_id = wandb.util.generate_id()
#     return job_id


# def setup_checkpoint_and_config_paths(cfg: DictConfig, job_id: str):
#     dirpath = Path(cfg.paths.checkpoint_dir) / "hourglass_vq" / job_id

#     # maybe load existing config, if resuming job
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

#     return dirpath


# @hydra.main(version_base=None, config_path="configs", config_name="train")
# def main(cfg: DictConfig):
#     # general set up
#     torch.set_float32_matmul_precision("medium")
#     job_id = setup_job_id(cfg)
#     dirpath = setup_checkpoint_and_config_paths(cfg, job_id)

#     # Data modules
#     datamodule = hydra.utils.instantiate(cfg.datamodule)
#     datamodule.setup()

#     # TODO: set up wandb

#     # launch distributed
#     world_size = torch.cuda.device_count()
#     mp.spawn(
#         train,
#         args=(world_size, cfg, datamodule),
#         nprocs=world_size,
#         join=True,
#     )


# if __name__ == "__main__":
#     main()
