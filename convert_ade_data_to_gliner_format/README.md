# ADE Dataset Splits

To create ADE dataset splits in the GLiNER format, run all the `create_*_splits.py` scripts first.

After creating the splits, you may want to chunk the instances into smaller portions according to the deberte-v3-large tokenizer. To do this, use the `chunk_jsons.py` script.