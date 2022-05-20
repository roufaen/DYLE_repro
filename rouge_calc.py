import re, sys
import numpy as np
from rouge import Rouge

def rouge_score(preds, refs):
        rouge_1, rouge_2, rouge_l = list(), list(), list()
        for pred, ref in zip(preds, refs):
            pred = re.sub('<\w+>', '', pred)
            ref = re.sub('<\w+>', '', ref)
            pred = ' '.join(pred)
            ref = ' '.join(ref)

            if len(ref) == 0 and len(pred) == 0:
                continue
            elif len(pred) == 0:
                rouge_1.append(0)
                rouge_2.append(0)
                rouge_l.append(0)
            else:
                score = Rouge().get_scores(refs=ref, hyps=pred)[0]
                rouge_1.append(score['rouge-1']['f'])
                rouge_2.append(score['rouge-2']['f'])
                rouge_l.append(score['rouge-l']['f'])

        return np.array(rouge_1).mean(), np.array(rouge_2).mean(), np.array(rouge_l).mean()


if __name__ == '__main__':
    output_fp = open(sys.argv[1], 'r')
    ref_fp = open(sys.argv[2], 'r')
    print(rouge_score(output_fp.read().split('\n'), ref_fp.read().split('\n')))
    output_fp.close()
    ref_fp.close()
