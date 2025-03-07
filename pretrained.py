from datasets import load_dataset

train_dataset = load_dataset("openslr/librispeech_asr", "clean", split = "Train.100", Streaming = True)
print(train_dataset)
