import data.cifar10 as cifar10
import data.nus_wide as nuswide
import data.imagenet as imagenet

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_data(dataset, root, num_query, num_train, batch_size, num_workers):
    """
    Load dataset.

    Args
        dataset(str): Dataset name.
        root(str): Path of dataset.
        num_query(int): Number of query data points.
        num_train(int): Number of training data points.
        num_workers(int): Number of loading data threads.

    Returns
        train_dataloader, query_dataloader, retrieval_dataloader(torch.utils.data.DataLoader): Data loader.
    """
    if dataset == 'cifar-10':
        train_dataloader, query_dataloader, retrieval_dataloader = cifar10.load_data(root,
                                                                                     num_query,
                                                                                     num_train,
                                                                                     batch_size,
                                                                                     num_workers,
                                                                                     )
    elif dataset == 'nus-wide-tc21':
        train_dataloader, query_dataloader, retrieval_dataloader = nuswide.load_data(21,
                                                                                     root,
                                                                                     num_query,
                                                                                     num_train,
                                                                                     batch_size,
                                                                                     num_workers,
                                                                                     )
    elif dataset == 'imagenet':
        train_dataloader, query_dataloader, retrieval_dataloader = imagenet.load_data(root, 
                                                                                      batch_size, 
                                                                                      num_workers,
                                                                                      )
    else:
        raise ValueError("Invalid dataset name!")

    return train_dataloader, query_dataloader, retrieval_dataloader
