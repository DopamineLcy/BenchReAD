from copy import deepcopy
from datasets import fundus
from torch.utils.data import DataLoader
from dataloaders.utlis import worker_init_fn_seed, BalancedBatchSampler

class initDataloader:
    DATASETS = {
        "fundus": fundus.Fundus,
    }

    @staticmethod
    def build(args, **kwargs):
        dataset_class = initDataloader.DATASETS.get(args.dataset)
        if not dataset_class:
            raise NotImplementedError

        # Create train and test sets
        train_set = dataset_class(args, train=True)
        test_set = dataset_class(args, train=False)

        # Create train and test loaders
        train_loader = DataLoader(
            train_set,
            batch_sampler=BalancedBatchSampler(args, train_set),
            worker_init_fn=worker_init_fn_seed,
            **kwargs
        )
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            worker_init_fn=worker_init_fn_seed,
            **kwargs
        )

        # Create reference loader
        temp_args = deepcopy(args)
        temp_args.batch_size = 5
        temp_args.nAnomaly = 0
        ref_set = dataset_class(temp_args, train=True, ref=True)
        ref_loader = DataLoader(
            ref_set,
            batch_size=temp_args.batch_size,
            shuffle=False,
            worker_init_fn=worker_init_fn_seed,
            **kwargs
        )

        return train_loader, test_loader, ref_loader