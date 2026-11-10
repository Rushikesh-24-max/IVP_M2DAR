"""Classification metrics, deliberately separate from temporal localization scoring."""
import math


def classification_metrics(labels, probabilities, classes=16):
    if not labels or len(labels) != len(probabilities):
        raise ValueError("Empty or mismatched predictions")
    confusion = [[0 for _ in range(classes)] for _ in range(classes)]
    top5 = 0
    for label, scores in zip(labels, probabilities):
        if type(label) is not int or not 0 <= label < classes or len(scores) != classes:
            raise ValueError("Invalid label/probability shape")
        if not all(math.isfinite(v) and 0 <= v <= 1.00001 for v in scores) or abs(sum(scores) - 1) > .001:
            raise ValueError("Expected finite softmax probabilities")
        ranking = sorted(range(classes), key=lambda i: scores[i], reverse=True)
        confusion[label][ranking[0]] += 1
        top5 += label in ranking[:min(5, classes)]
    per_class = []
    for i in range(classes):
        tp = confusion[i][i]
        support = sum(confusion[i])
        predicted = sum(row[i] for row in confusion)
        precision = tp / predicted if predicted else 0.
        recall = tp / support if support else 0.
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.
        per_class.append(dict(class_id=i, support=support, precision=precision, recall=recall, f1=f1))
    return dict(samples=len(labels), accuracy=sum(confusion[i][i] for i in range(classes)) / len(labels),
                top5_accuracy=top5 / len(labels), macro_f1=sum(v["f1"] for v in per_class) / classes,
                balanced_accuracy=sum(v["recall"] for v in per_class) / classes,
                per_class=per_class, confusion_matrix=confusion)
