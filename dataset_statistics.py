import json
from matplotlib import pyplot as plt
import numpy as np


if __name__ == '__main__':
    len_of_article, len_of_summary, num_of_oracle = list(), list(), list()
    for file in ['../datasets/CNewSum/train.simple.label.jsonl', '../datasets/CNewSum/dev.simple.label.jsonl']:
        with open(file, 'r') as fp:
            for line in fp.readlines():
                line_json = json.loads(line)
                for sentence in line_json['article']:
                    len_of_article.append(len(sentence))
                len_of_summary.append(len(line_json['summary']))
                num_of_oracle.append(len(line_json['label']))
    len_of_article, len_of_summary, num_of_oracle = np.array(len_of_article), np.array(len_of_summary), np.array(num_of_oracle)
    print(len_of_article.mean(), len_of_article.max(), len_of_summary.mean(), len_of_summary.max(), num_of_oracle.mean(), num_of_oracle.max())

    plt.cla()
    plt.hist(len_of_article, bins=50, range=(0, 1000), rwidth=0.8)
    plt.title('len_of_article')
    plt.savefig('len_of_article.png')
    plt.cla()
    plt.hist(len_of_summary, bins=50, range=(0, 220), rwidth=0.8)
    plt.title('len_of_summary')
    plt.savefig('len_of_summary.png')
    plt.cla()
    plt.hist(num_of_oracle, bins=50, range=(0, 8), rwidth=0.8)
    plt.title('num_of_oracle')
    plt.savefig('num_of_oracle.png')
