import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


parser = argparse.ArgumentParser(description='Time Analysis')
parser.add_argument('--log-dir', type=str, default='./logs/time_analysis/cuda')
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--in-features', type=int, default=128)
parser.add_argument('--out-features', type=int, default=10)
parser.add_argument('--samples', type=int, default=10)

if __name__ == '__main__':
    suptitlesize = 20
    titlesize = 18
    labelsize = 14
    legendsize = 12
    ticksize = 12
    
    args = parser.parse_args()
    log_f_name = args.log_dir + '/cuda_batch' + str(args.batch_size) + '_in' + str(args.in_features) + '_out' + str(args.out_features) + '_samples' + str(args.samples) + '.csv'
    
    df = pd.read_csv(log_f_name)
    print(df)

    plt.figure(figsize=(4, 3))
    sns.set(style="whitegrid")
    plt.errorbar(df['L'], df['dense_mean'], yerr=df['dense_std'], label='Dense', fmt='-o')
    plt.errorbar(df['L'], df['sparse_mean'], yerr=df['sparse_std'], label='Sparse (ours)', fmt='-o')
    plt.yscale('log')
    plt.xlabel('L (Dyadic level)', fontsize=labelsize)
    plt.ylabel('Time (ms, log scale)', fontsize=labelsize)
    # plt.title(f'Time Analysis (Batch Size={args.batch_size}, In Features={args.in_features}, Out Features={args.out_features}, Samples={args.samples})')
    plt.legend(fontsize=legendsize)
    plt.tight_layout()
    plt.savefig(args.log_dir + '/cuda_time_analysis_batch' + str(args.batch_size) + '_in' + str(args.in_features) + '_out' + str(args.out_features) + '_samples' + str(args.samples) + '.pdf')
    plt.show()