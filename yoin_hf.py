from datasets import load_dataset
import os

def save_ds(partition: str, path: str = "./Data"):
    os.makedirs(path, exist_ok = True)
    ds = load_dataset("openslr/librispeech_asr", "clean", split =f'{partition}' )
    ds.save_to_disk(path)
    print(f'saved ds to {path}')

if __name__ == "__main__":
    save_ds('Train.100')
