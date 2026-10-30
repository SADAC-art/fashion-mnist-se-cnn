import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from scipy.stats import binomtest


def classification_metrics(target, prediction, classes):
    precision, recall, f1, _ = precision_recall_fscore_support(target, prediction, average='macro', zero_division=0)
    return {'accuracy': float(accuracy_score(target, prediction)), 'precision_macro': float(precision),
            'recall_macro': float(recall), 'f1_macro': float(f1)}, classification_report(target, prediction, target_names=classes, output_dict=True, zero_division=0), confusion_matrix(target, prediction, labels=list(range(10)))


def paired_comparison(target, base, method, seed=9901):
    # 同一批测试样本上的配对差异；bootstrap 不代表跨训练种子置信区间。
    a, b = base == target, method == target
    gain = int((~a & b).sum()); loss = int((a & ~b).sum()); n = len(target)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n, [gain/n, loss/n, 1-(gain+loss)/n], size=10000)
    boot = (draws[:, 0] - draws[:, 1]) / n
    return {'gain_n': gain, 'loss_n': loss, 'delta_accuracy': float(b.mean()-a.mean()),
            'bootstrap_ci95': np.quantile(boot, [.025, .975]).tolist(), 'bootstrap_repeats': 10000,
            'mcnemar_exact_p': float(binomtest(gain, gain+loss, .5).pvalue) if gain+loss else 1.0}
