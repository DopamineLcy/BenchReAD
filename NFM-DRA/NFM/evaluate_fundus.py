# %%
import contextlib
import logging
import os
import sys

import click
import numpy as np
import torch


import src.patchcore as patchcore
import src.patchcore.metrics
import src.patchcore.patchcore
import src.patchcore.sampler
import src.patchcore.utils
import src.patchcore.common
import gc


LOGGER = logging.getLogger(__name__)

_DATASETS = {"fundus": ["patchcore.datasets.fundus", "FundusDataset"]}

# %%
gpu=[3]
device = patchcore.utils.set_torch_device(gpu)
device_context = (
    torch.cuda.device("cuda:{}".format(device.index))
    if "cuda" in device.type.lower()
    else contextlib.suppress()
)

# %%
def dataset(
    name="fundus", split=None, data_path=".", subdatasets=("fundus",), batch_size=1, resize=224, imagesize=224, num_workers=8, augment=False
):
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders_iter(seed):
        for subdataset in subdatasets:
            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=split,
                seed=seed,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            test_dataloader.name = name
            if subdataset is not None:
                test_dataloader.name += "_" + subdataset

            dataloader_dict = {"testing": test_dataloader}

            yield dataloader_dict

    return get_dataloaders_iter

# %%
def patch_core_loader(patch_core_paths=["NFM-DRA/NFM/results/Fundus_Results_224_224_0.1/IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0/models/fundus_fundus"], faiss_on_gpu=True, faiss_num_workers=8):
    def get_patchcore_iter(device):
        for patch_core_path in patch_core_paths:
            loaded_patchcores = []
            gc.collect()
            n_patchcores = len(
                [x for x in os.listdir(patch_core_path) if ".faiss" in x]
            )
            if n_patchcores == 1:
                nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)
                patchcore_instance = patchcore.patchcore.PatchCore(device)
                patchcore_instance.load_from_path(
                    load_path=patch_core_path, device=device, nn_method=nn_method
                )
                loaded_patchcores.append(patchcore_instance)
            else:
                for i in range(n_patchcores):
                    nn_method = patchcore.common.FaissNN(
                        faiss_on_gpu, faiss_num_workers
                    )
                    patchcore_instance = patchcore.patchcore.PatchCore(device)
                    patchcore_instance.load_from_path(
                        load_path=patch_core_path,
                        device=device,
                        nn_method=nn_method,
                        prepend="Ensemble-{}-{}_".format(i + 1, n_patchcores),
                    )
                    loaded_patchcores.append(patchcore_instance)

            yield loaded_patchcores

    return get_patchcore_iter

dataset_info = _DATASETS["fundus"]
dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

for split in [dataset_library.DatasetSplit.VAL, dataset_library.DatasetSplit.RIADD, dataset_library.DatasetSplit.JSIEC]:
    seed = 0
    dataloader_iter = dataset(split=split)
    dataloader_iter = dataloader_iter(seed)
    patchcore_iter = patch_core_loader()
    patchcore_iter = patchcore_iter(device)
    n_dataloaders = 1
    n_patchcores = 1
    for dataloader_count, dataloaders in enumerate(dataloader_iter):
        print(dataloader_count)
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["testing"].name, dataloader_count + 1, n_dataloaders
            )
        )

        patchcore.utils.fix_seeds(seed, device)

        dataset_name = dataloaders["testing"].name

        with device_context:

            torch.cuda.empty_cache()
            if dataloader_count < n_patchcores:
                PatchCore_list = next(patchcore_iter)

            aggregator = {"scores": [], "segmentations": []}
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Embedding test data with models ({}/{})".format(
                        i + 1, len(PatchCore_list)
                    )
                )
                scores, segmentations, labels_gt, masks_gt = PatchCore.predict(
                    dataloaders["testing"]
                )
                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)

            scores = np.array(aggregator["scores"])
            min_scores = scores.min(axis=-1).reshape(-1, 1)
            max_scores = scores.max(axis=-1).reshape(-1, 1)
            scores = (scores - min_scores) / (max_scores - min_scores)
            scores = np.mean(scores, axis=0)

            anomaly_labels = [
                x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
            ]


        # %%
        y_prob = scores
        y_true = np.array(anomaly_labels).astype(int)

        # %%
        # write y_true and p_prob to results.txt
        with open(os.path.join("NFM-DRA/NFM/results/Fundus_Results_224_224_0.1/", f'{split.value}_results.txt'), 'w') as f:
            for i in range(len(y_true)):
                f.write(f'{y_true[i]}\t{y_prob[i]}\n')


