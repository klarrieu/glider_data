import os
import pandas as pd
import numpy as np

# directory with ascii files (.dat) and headers (.m) parsed from binaries with dbd2asc.py
ascii_dir = os.path.join(os.path.dirname(__file__), 'data\\glider\\ascii')

# list of dataframes for each file
dfs = []


def read_m_header(filename):
    # read header from matlab (*.m) file
    cols = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('\'segment_filenames\''):
                break
            # lines that are part of the struct() args begin with apostrophe (')
            if line.startswith('\''):
                var, i = line.split(',')[:2]
                var = var.replace('\'', '')
                i = int(i)
                cols.append((i, var))
    colnames = [x for _, x in sorted(cols)]
    return colnames


i = 0
for dir, subdirs, files in os.walk(ascii_dir):
    for f in files:
        if f.endswith('.dat'):
            print(i)
            i += 1

            print('loading %s' % f)
            filename = os.path.join(dir, f)
            header = read_m_header(filename.replace('.dat', '.m'))
            df = pd.read_csv(filename, sep=' ', header=None, index_col=False, names=header)
            dfs.append(df)

# concatenate all data into one dataframe
print('concatenating all data...')
df = pd.concat(dfs, sort=True)
# print('writing to master csv...')
# df.to_csv('LakeSuperior.csv', index=False)
print('writing science data to csv...')
df_sci = df[[col for col in df.columns if col.startswith('sci_')]]
df_sci.to_csv('LakeSuperior_sci.csv', index=False)
