import os
import argparse
import feature_extraction.feature_extraction as feature_extraction
import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--input_dir", type=str, required=True)
    args.add_argument("--output_path", type=str, required=True)

    df = pd.DataFrame()

    for f in tqdm(os.listdir(args.input_dir)):
        if f.endswith('.nii.gz'):
            path = os.path.join(args.input_dir, f)
            sub_id = f.split('_')[0]
            out = feature_extraction.analyze_vessels(path)
            out['sub_id'] = sub_id
            df = df.append(out, ignore_index=True)
