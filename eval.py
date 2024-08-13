from library.evaluation import *
from library.utils import attachment2span, read_attachment_file

prediction_path = 'Baselines/Dependency_Parser_Aggregation/output.txt'
gold_path = "outputs/Gold/te.txt"


def f1_eval(preds, refs):
    preds_spans = [attachment2span(a) for a in preds]
    refs_spans = [attachment2span(a) for a in refs]
    return f"CorpusF1: {corpus_f1(preds_spans, refs_spans)}"


def acc_eval(preds, refs):
    return f"CorpusUAS: {corpus_acc(preds, refs, add_zeros=True)}"


def eval(preds, refs):
    return f1_eval(preds, refs) + "\t" + acc_eval(preds, refs)


prediction = read_attachment_file(prediction_path)
gold = read_attachment_file(gold_path)
print(eval(prediction, gold))
