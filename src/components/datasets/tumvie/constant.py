DATA_SPLIT = {
    "default":
    {
        "train": ["train/seq0", "train/seq1", "train/seq2"],
        "valid": ["valid/seq0", "valid/seq1"],
        "test": ["test/seq0"],  #"test/seq1", "test/seq2", "test/seq3", "test/seq4", "test/seq5", "test/seq6", "test/seq7"
        "none": [],
    },
    "unitreego":
    {
        "train": ["train/seq0", "train/seq2", "train/seq3", "train/seq4", "train/seq5"],
        "valid": ["valid/seq0", "valid/seq2", "valid/seq3", "valid/seq4"],
        "test": ["test/seq14"], # ["test/seq14", "test/seq12", "test/seq22"],
        "none": [],
    }
}
