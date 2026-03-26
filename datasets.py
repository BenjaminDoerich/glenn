import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import __init__
import sys
import lightning as L
import torch
import math

class RefreshableInMemoryDataset(Dataset):

    def __init__(self, total_samples: int, base_seed: int = 0):
        self.total_samples = total_samples
        self.base_seed = base_seed
        self._generate_data()


    def _generate_data(self):
        g = torch.Generator()
        g.manual_seed(self.base_seed)
        data = torch.rand((self.total_samples, 3), generator=g)

        data[:, 2] = 0.05 + 0.95 * data[:, 2]
        self.data = data


    def refresh(self, new_seed: int):
        self.base_seed = new_seed
        self._generate_data()


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        return self.data[idx]



class ValidationMeshgridDataset(Dataset):

    def __init__(self, n_xy: int = 32, z_values=(8, 16, 20, 24)):
        xs = torch.linspace(0, 1, n_xy)
        ys = torch.linspace(0, 1, n_xy)
        self.z_values = z_values
        self.data_by_z = {}

        for z in z_values:
            xv, yv = torch.meshgrid(xs, ys, indexing="ij")
            zv = torch.full_like(xv, float(z))
            pts = torch.stack([xv, yv, zv], axis=-1)
            tensor_pts = torch.tensor(pts.reshape(-1, 3))
            self.data_by_z[z] = tensor_pts


    def get_subset(self, z):
        return self.data_by_z[z]



class ValidationMeshgridDiskDataset(Dataset):

    def __init__(self, n_xy: int = 32, z_values=(8, 16, 20, 24)):

        g = torch.Generator()
        g.manual_seed(0)

        u = torch.rand(n_xy**2, generator=g)
        r = torch.sqrt(u)
        theta = 2.0 * torch.pi * torch.rand(
            n_xy**2, generator=g
        )

        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        self.z_values = z_values
        self.data_by_z = {}

        for z in z_values:
            zv = torch.full_like(x, float(z))
            self.data_by_z[z] = torch.stack([x, y, zv], dim=1)


    def get_subset(self, z):
        return self.data_by_z[z]



class RefreshableInMemoryDatasetSingleKappa(Dataset):

    def __init__(self, total_samples: int, base_seed: int = 0):
        self.total_samples = total_samples
        self.base_seed = base_seed
        self._generate_data()


    def _generate_data(self):
        g = torch.Generator()
        g.manual_seed(self.base_seed)
        data = torch.rand((self.total_samples, 2), generator=g)
        self.data = data


    def refresh(self, new_seed: int):
        self.base_seed = new_seed
        self._generate_data()


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        return self.data[idx]

    

class RefreshableInMemoryDataModuleSingleKappa(L.LightningDataModule):

    def __init__(
        self,
        global_train_batch_size: int = 8192,
        total_train_samples: int = 10_000_000,
        base_seed: int = 0,
        num_workers: int = 4,
        refresh_every_epochs: int = 1,
    ):
        super().__init__()
        self.global_train_batch_size = global_train_batch_size
        self.total_train_samples = total_train_samples
        self.base_seed = base_seed
        self.num_workers = num_workers
        self.refresh_every_epochs = refresh_every_epochs


    def setup(self, stage=None):
        if hasattr(self, "trainer") and self.trainer is not None:
            num_devices = getattr(self.trainer, "num_devices", 1)
            num_nodes = getattr(self.trainer, "num_nodes", 1)
            world_size = getattr(self.trainer, "world_size", num_devices * num_nodes)
        elif torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            num_devices = world_size
            num_nodes = 1
        else:
            world_size, num_devices, num_nodes = 1, 1, 1

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        self.per_device_train_batch = max(1, self.global_train_batch_size // world_size)

        if rank == 0:
            print("===== RefreshableInMemoryDataModule Setup =====")
            print(f"num_nodes={num_nodes}, num_devices={num_devices}, world_size={world_size}")
            print(f"global_train_batch_size={self.global_train_batch_size} -> per-device {self.per_device_train_batch}")
            print(f"num_workers={self.num_workers}")
            print(f"refresh_every_epochs={self.refresh_every_epochs}")
            print("===============================================")

        self.train_dataset = RefreshableInMemoryDatasetSingleKappa(self.total_train_samples, self.base_seed)


    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset, shuffle=True, drop_last=True)
        return DataLoader(
            self.train_dataset,
            batch_size=self.per_device_train_batch,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )



class RefreshableInMemoryDiskDataset(Dataset):

    def __init__(self, total_samples: int, base_seed: int = 0):
        self.total_samples = total_samples
        self.base_seed = base_seed
        self._generate_data()


    def _generate_data(self):
        g = torch.Generator()
        g.manual_seed(self.base_seed)

        u = torch.rand(self.total_samples, generator=g)
        r = torch.sqrt(u)
        theta = 2.0 * math.pi * torch.rand(
            self.total_samples, generator=g
        )

        x = r * torch.cos(theta)
        y = r * torch.sin(theta)

        z = 0.05 + 0.95 * torch.rand(
            self.total_samples, generator=g
        )

        self.data = torch.stack([x, y, z], dim=1)


    def refresh(self, new_seed: int):

        self.base_seed = new_seed
        self._generate_data()


    def __len__(self):
        return self.data.shape[0]


    def __getitem__(self, idx):
        return self.data[idx]



class LShapeDataset(Dataset):
    def __init__(self, total_samples: int, base_seed: int = 0):
        self.total_samples = total_samples
        self.samples_per_square = total_samples//3
        self.base_seed = base_seed
        self._generate_data()


    def _generate_data(self):
        g = torch.Generator()
        g.manual_seed(self.base_seed)

        data_1 = torch.tensor((0.5, 0.5, 1.0))*torch.rand((self.samples_per_square, 3), generator=g)
        data_2 = torch.rand((self.samples_per_square, 3), generator=g)
        data_2 = torch.tensor((0.5, 0.5, 1.0))*data_2 + torch.tensor((0.5, 0.0, 0.0))
        data_3 = torch.rand((self.samples_per_square, 3), generator=g)
        data_3 = torch.tensor((0.5, 0.5, 1.0))*data_3 + torch.tensor((0.0, 0.5, 0.0))

        data = torch.cat([data_1, data_2,data_3])
        data[:, 2] = 0.05 + 0.95 * data[:, 2]
        self.data = data


    def refresh(self, new_seed: int):
        self.base_seed = new_seed
        self._generate_data()


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        return self.data[idx]



class ValidationLShapeDataset(Dataset):

    def __init__(self, n_xy: int = 32, z_values=(8, 16, 20, 24)):
        xs = torch.linspace(0, 1, n_xy)
        ys = torch.linspace(0, 1, n_xy)
        self.z_values = z_values
        self.data_by_z = {}
        xv, yv = torch.meshgrid(xs, ys, indexing="ij")
        xyv = torch.stack([xv.reshape(-1),yv.reshape(-1)], axis=-1)

        data_1 = torch.tensor((0.5, 0.5))*xyv
        data_2 = xyv
        data_2 = torch.tensor((0.5, 0.5))*data_2 + torch.tensor((0.5, 0.0))
        data_3 = xyv
        data_3 = torch.tensor((0.5, 0.5))*data_3 + torch.tensor((0.0, 0.5))

        data = torch.cat([data_1, data_2,data_3])

        
        for z in z_values:
            zv = torch.full_like(data[:,[0]], float(z))
            pts = torch.cat([data, zv], dim=1)

            self.data_by_z[z] = pts


    def get_subset(self, z):
        return self.data_by_z[z]



class ValidationDiskMeshgridDataset(Dataset):

    def __init__(self, n_xy: int = 64, z_values=(8, 16, 20, 24)):

        xs = torch.linspace(-1.0, 1.0, n_xy)
        ys = torch.linspace(-1.0, 1.0, n_xy)

        xv, yv = torch.meshgrid(xs, ys, indexing="ij")

        mask = xv**2 + yv**2 <= 1.0

        x_disk = xv[mask]
        y_disk = yv[mask]

        self.data_by_z = {}

        for z in z_values:
            z_tensor = torch.full_like(x_disk, float(z))
            pts = torch.stack([x_disk, y_disk, z_tensor], dim=1)
            self.data_by_z[z] = pts


    def get_subset(self, z):

        return self.data_by_z[z]



class RefreshableInMemoryDataModule(L.LightningDataModule):

    def __init__(
        self,
        global_train_batch_size: int = 8192,
        global_val_batch_size: int = 4096,
        total_train_samples: int = 10_000_000,
        val_grid_size: int = 32,
        val_z_values=(8, 16, 20, 24),
        base_seed: int = 0,
        num_workers: int = 4,
        refresh_every_epochs: int = 1,
        domain: str = "unit_square",
    ):
        super().__init__()
        self.global_train_batch_size = global_train_batch_size
        self.global_val_batch_size = global_val_batch_size
        self.total_train_samples = total_train_samples
        self.val_grid_size = val_grid_size
        self.val_z_values = val_z_values
        self.base_seed = base_seed
        self.num_workers = num_workers
        self.refresh_every_epochs = refresh_every_epochs
        self.domain = domain


    def setup(self, stage=None):
        if hasattr(self, "trainer") and self.trainer is not None:
            num_devices = getattr(self.trainer, "num_devices", 1)
            num_nodes = getattr(self.trainer, "num_nodes", 1)
            world_size = getattr(self.trainer, "world_size", num_devices * num_nodes)
        elif torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            num_devices = world_size
            num_nodes = 1
        else:
            world_size, num_devices, num_nodes = 1, 1, 1

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        self.per_device_train_batch = max(1, self.global_train_batch_size // world_size)
        self.per_device_val_batch = max(1, self.global_val_batch_size // world_size)

        if rank == 0:
            print("===== RefreshableInMemoryDataModule Setup =====")
            print(f"num_nodes={num_nodes}, num_devices={num_devices}, world_size={world_size}")
            print(f"global_train_batch_size={self.global_train_batch_size} -> per-device {self.per_device_train_batch}")
            print(f"global_val_batch_size={self.global_val_batch_size} -> per-device {self.per_device_val_batch}")
            print(f"num_workers={self.num_workers}")
            print(f"refresh_every_epochs={self.refresh_every_epochs}")
            print("===============================================")

        if self.domain == "circle":
            self.train_dataset = RefreshableInMemoryDiskDataset(self.total_train_samples, self.base_seed)
            val_dataset = ValidationMeshgridDiskDataset(self.val_grid_size, self.val_z_values)
        elif self.domain == "Lshape":
            self.train_dataset = LShapeDataset(self.total_train_samples, self.base_seed)
            val_dataset = ValidationLShapeDataset(self.val_grid_size, self.val_z_values)
        else:
            self.train_dataset = RefreshableInMemoryDataset(self.total_train_samples, self.base_seed)
            val_dataset = ValidationMeshgridDataset(self.val_grid_size, self.val_z_values)

        self.val_datasets = {z: val_dataset.get_subset(z) for z in self.val_z_values}


    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset, shuffle=True, drop_last=True)
        return DataLoader(
            self.train_dataset,
            batch_size=self.per_device_train_batch,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    

    def val_dataloader(self):
        loaders = []
        for z, subset in self.val_datasets.items():
            sampler = DistributedSampler(subset, shuffle=False, drop_last=False)
            loaders.append(
                DataLoader(
                    subset,
                    batch_size=self.per_device_val_batch,
                    sampler=sampler,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )
            )
        return loaders



class RefreshableInMemoryRefinementDataset(Dataset):

    def __init__(self, total_samples: int, base_seed: int = 0, kappa = 100):
        self.total_samples = total_samples
        self.base_seed = base_seed
        self.kappa = kappa
        self._generate_data()


    def _generate_data(self):
        g = torch.Generator()
        g.manual_seed(self.base_seed)

        xy = torch.rand((self.total_samples, 2), generator=g)
        z = torch.full((self.total_samples, 1), self.kappa, dtype=xy.dtype, device=xy.device)
        self.data = torch.cat([xy, z], dim=1)


    def refresh(self, new_seed: int):
        self.base_seed = new_seed
        self._generate_data()


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        return self.data[idx]



class RefreshableInMemoryRefinementDiskDataset(Dataset):
    def __init__(self, total_samples: int, base_seed: int = 0, kappa = 100):
        self.total_samples = total_samples
        self.base_seed = base_seed
        self.kappa = kappa
        self._generate_data()


    def _generate_data(self):
        g = torch.Generator()
        g.manual_seed(self.base_seed)

        u = torch.rand(self.total_samples, generator=g)
        r = torch.sqrt(u)
        theta = 2.0 * math.pi * torch.rand(
            self.total_samples, generator=g
        )
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        z = torch.full((self.total_samples, 1), self.kappa, dtype=x.dtype, device=y.device)

        self.data = torch.stack([x, y, z], dim=1)


    def refresh(self, new_seed: int):

        self.base_seed = new_seed
        self._generate_data()


    def __len__(self):
        return self.data.shape[0]


    def __getitem__(self, idx):
        return self.data[idx]



class RefreshableInMemoryRefinementDataModule(L.LightningDataModule):

    def __init__(
        self,
        global_train_batch_size: int = 8192,
        total_train_samples: int = 10_000_000,
        global_val_batch_size: int = 4096,
        val_grid_size: int = 32,
        val_z_values=(8, 16, 20, 24),
        base_seed: int = 0,
        num_workers: int = 4,
        refresh_every_epochs: int = 1,
        domain: str = "unit_square",
        kappa = 100,

    ):
        super().__init__()
        self.global_train_batch_size = global_train_batch_size
        self.global_val_batch_size = global_val_batch_size
        self.val_grid_size = val_grid_size
        self.total_train_samples = total_train_samples
        self.base_seed = base_seed
        self.num_workers = num_workers
        self.refresh_every_epochs = refresh_every_epochs
        self.domain = domain
        self.kappa = kappa
        self.val_z_values=val_z_values
        

    def setup(self, stage=None):
        if hasattr(self, "trainer") and self.trainer is not None:
            num_devices = getattr(self.trainer, "num_devices", 1)
            num_nodes = getattr(self.trainer, "num_nodes", 1)
            world_size = getattr(self.trainer, "world_size", num_devices * num_nodes)
        elif torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            num_devices = world_size
            num_nodes = 1
        else:
            world_size, num_devices, num_nodes = 1, 1, 1

        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

        self.per_device_train_batch = max(1, self.global_train_batch_size // world_size)
        self.per_device_val_batch = max(1, self.global_val_batch_size // world_size)

        if rank == 0:
            print("===== RefreshableInMemoryRefinementDataModule Setup =====")
            print(f"num_nodes={num_nodes}, num_devices={num_devices}, world_size={world_size}")
            print(f"global_train_batch_size={self.global_train_batch_size} -> per-device {self.per_device_train_batch}")
            print(f"global_val_batch_size={self.global_val_batch_size} -> per-device {self.per_device_val_batch}")
            print(f"num_workers={self.num_workers}")
            print(f"refresh_every_epochs={self.refresh_every_epochs}")
            print("===============================================")

        if self.domain == "circle":
            self.train_dataset = RefreshableInMemoryRefinementDiskDataset(self.total_train_samples, self.base_seed, kappa = self.kappa)
        else:
            self.train_dataset = RefreshableInMemoryRefinementDataset(self.total_train_samples, self.base_seed, kappa = self.kappa)
            val_dataset = ValidationMeshgridDataset(self.val_grid_size, self.val_z_values)

        self.val_datasets = {z: val_dataset.get_subset(z) for z in self.val_z_values}


    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset, shuffle=True, drop_last=True)
        return DataLoader(
            self.train_dataset,
            batch_size=self.per_device_train_batch,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    

    def val_dataloader(self):
        loaders = []
        for z, subset in self.val_datasets.items():
            sampler = DistributedSampler(subset, shuffle=False, drop_last=False)
            loaders.append(
                DataLoader(
                    subset,
                    batch_size=self.per_device_val_batch,
                    sampler=sampler,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )
            )
        return loaders



class RefreshLogger(L.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        dm = trainer.datamodule
        epoch = trainer.current_epoch

        if (epoch + 1) % dm.refresh_every_epochs == 0:
            new_seed = dm.base_seed + (epoch + 1)
            dm.train_dataset.refresh(new_seed=new_seed)

            is_rank0 = True
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                is_rank0 = (torch.distributed.get_rank() == 0)

            if is_rank0:
                print(f"[RefreshLogger] Epoch {epoch} -> refreshed with base_seed={new_seed}",
                        file=sys.stderr, flush=True)

                if trainer.logger is not None:
                    trainer.logger.log_metrics({"refresh_seed": new_seed}, step=epoch)
