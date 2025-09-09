import pickle

# 假设文件名是 snapshot.pkl
with open(
    "/home/deadpool/Playground/CS336/assignment1-basics/tests/_snapshots/test_train_bpe_special_tokens.pkl",
    "rb",
) as f:
    data = pickle.load(f)

with open(
    "/home/deadpool/Playground/CS336/assignment1-basics/cs336_basics/merges_output.txt",
    "w",
) as f:
    merges = data["merges"]
    for idx, merge in enumerate(merges):
        f.write(f"id: {idx}: {merge}\n")
