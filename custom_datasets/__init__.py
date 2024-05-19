from datasets import Dataset as HFDataset, Features, Image as HFImage, Value
import multiprocessing as mp

from .my_datasets import PNDBUfesDataset

def parallel_generator_fn(args):
    dataset_class, root, csv_train, csv_test, train, start_idx, end_idx = args
    dataset = dataset_class(root, csv_train, csv_test, train)
    results = []
    for i in range(start_idx, end_idx):
        results.append(dataset[i])
    return results

def split_indices(num_indices, num_splits):
    split_size = num_indices // num_splits
    indices = [(i * split_size, (i + 1) * split_size if i != num_splits - 1 else num_indices) for i in range(num_splits)]
    return indices

def get_dataset_class(dataset_name):
    if dataset_name == "P-NDB-UFES":
        return PNDBUfesDataset
    else:
        raise NotImplementedError(f"There isn't such dataset implemented: {dataset_name}")

def get_custom_dataset(dataset_name, root, csv_train, csv_test):
    # Number of processes for parallel generation
    num_processes = mp.cpu_count()

    # Split the indices for parallel processing
    dataset_class = get_dataset_class(dataset_name)
    dataset = dataset_class(root, csv_train, csv_test, train=True)
    indices = split_indices(len(dataset), num_processes)

    # Create Hugging Face Dataset from parallel generator function
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(
            parallel_generator_fn,
            [(dataset_class, root, csv_train, csv_test, True, start, end) for start, end in indices]
        )
        flattened_results = [item for sublist in results for item in sublist]

    hf_dataset = HFDataset.from_generator(
        lambda: iter(flattened_results),
        features=Features(
            {
                "image": HFImage(),
                "lesion": Value("string")
            }
        )
    )
    
    return hf_dataset