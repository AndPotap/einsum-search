import os
from tqdm import tqdm
import numpy as np
import pickle
from datasets import load_dataset

out_dir = './open_small'
num_proc = 64
num_proc_load_dataset = num_proc

# Define the list of allowed tokens
allowed_tokens = [
    '\n', ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7',
    '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
    'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
    'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~'
]

# Create a mapping from tokens to integers
stoi = {token: i for i, token in enumerate(allowed_tokens)}
itos = {i: token for i, token in enumerate(allowed_tokens)}


def encode(text):
    return [stoi[char] for char in text if char in allowed_tokens]


def decode(ids):
    return ''.join([itos[i] for i in ids])


if __name__ == '__main__':
    dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

    def filter_example(example):
        return all(char in allowed_tokens for char in example['text'])

    # Filter the dataset
    filtered_dataset = split_dataset.filter(
        filter_example,
        desc="filtering the splits",
        num_proc=num_proc,
    )

    # Print the percentage of documents left after filtering
    print(
        f"Percentage of documents left after filtering: {len(filtered_dataset['train']) / len(split_dataset['train']) * 100:.2f}%"
    )

    def process(example):
        ids = encode(example['text'])
        ids.append(stoi['\n'])  # Add end of text token
        out = {'ids': ids, 'len': len(ids)}
        return out

    # Tokenize the filtered dataset
    tokenized = filtered_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # Concatenate all the ids in each dataset into one large file for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        print(f'{split}: {arr_len}')
        filename = os.path.join(os.path.dirname(out_dir), f'{split}.bin')
        dtype = np.uint8  # Assuming the number of allowed tokens is less than 256
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len, ))
        total_batches = 512
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx:idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

    # Save the meta information
    meta = {
        'vocab_size': len(allowed_tokens),
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(os.path.dirname(out_dir), 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)