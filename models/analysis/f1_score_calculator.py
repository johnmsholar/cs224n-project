from  sklearn.metrics import f1_score

LABELS =  [0, 1, 2]
LABEL_MAPPING = {
    'agree': 0,
    'disagree': 1,
    'discuss': 2,
    'unrelated': 3,
}

def consruct_lists(distr):
    gold = []
    preds = []

    # Agree 
    total_agree = distr["agree"]["agree"] + distr["agree"]["disagree"] + distr["agree"]["discuss"]
    gold.extend([LABEL_MAPPING["agree"]] * total_agree)
    preds.extend([LABEL_MAPPING["agree"]] * distr["agree"]["agree"] + [LABEL_MAPPING["disagree"]] * distr["agree"]["disagree"] + [LABEL_MAPPING["discuss"]] * distr["agree"]["discuss"])

    # Disagree
    total_disagree = distr["disagree"]["agree"] + distr["disagree"]["disagree"] + distr["disagree"]["discuss"]
    gold.extend([LABEL_MAPPING["disagree"]] * total_disagree)
    preds.extend([LABEL_MAPPING["agree"]] * distr["disagree"]["agree"] + [LABEL_MAPPING["disagree"]] * distr["disagree"]["disagree"] + [LABEL_MAPPING["discuss"]] * distr["disagree"]["discuss"])
  
    # Discuss
    total_discuss = distr["discuss"]["agree"] + distr["discuss"]["disagree"] + distr["discuss"]["discuss"]
    gold.extend([LABEL_MAPPING["discuss"]] * total_discuss)
    preds.extend([LABEL_MAPPING["agree"]] * distr["discuss"]["agree"] + [LABEL_MAPPING["disagree"]] * distr["discuss"]["disagree"] + [LABEL_MAPPING["discuss"]] * distr["discuss"]["discuss"])
  
    return gold, preds


if __name__ == '__main__':

    # Conditional LSTM
    conditional_lstm = {
        "agree": {
            "agree": 139,
            "disagree":  0,
            "discuss": 516,        
        },
        "disagree": {
            "agree": 34,
            "disagree":  0,
            "discuss": 127, 
        },
        "discuss": {
            "agree": 77,
            "disagree":  0,
            "discuss": 1896, 
        }
    }
    gold, preds = consruct_lists(conditional_lstm)
    print "Conditional LSTM Micro F1 Score: {}".format(f1_score(gold, preds, labels=LABELS, average='micro'))

    # Headline to Article Global Attention
    global_attention = {
        "agree": {
            "agree": 416,
            "disagree":  11,
            "discuss": 228,        
        },
        "disagree": {
            "agree": 68,
            "disagree":  17,
            "discuss": 76, 
        },
        "discuss": {
            "agree": 271,
            "disagree":  36,
            "discuss": 1666, 
        }
    }
    gold, preds = consruct_lists(global_attention)
    print "Conditional Global Attention Micro F1 Score: {}".format(f1_score(gold, preds, labels=LABELS, average='micro'))

    # Word By Word Attention
    wbw_attention = {
        "agree": {
            "agree": 438,
            "disagree":  6,
            "discuss": 211,        
        },
        "disagree": {
            "agree": 70,
            "disagree":  4,
            "discuss": 87, 
        },
        "discuss": {
            "agree": 261,
            "disagree":  11,
            "discuss": 1701, 
        }
    }
    gold, preds = consruct_lists(wbw_attention)
    print "Conditional WBW Micro F1 Score: {}".format(f1_score(gold, preds, labels=LABELS, average='micro'))

    # Bidirectional Global Attention
    bi_att = {
        "agree": {
            "agree": 459,
            "disagree":  0,
            "discuss": 196,        
        },
        "disagree": {
            "agree": 88,
            "disagree":  0,
            "discuss": 73, 
        },
        "discuss": {
            "agree": 266,
            "disagree":  0,
            "discuss": 1707, 
        }
    }
    gold, preds = consruct_lists(bi_att)
    print "Conditional WBW Micro F1 Score: {}".format(f1_score(gold, preds, labels=LABELS, average='micro'))

    # Bidirectional Attention Bidirectional conditional encoding lstm
    bi_att_bi_condition = {
        "agree": {
            "agree": 444,
            "disagree":  30,
            "discuss": 181,        
        },
        "disagree": {
            "agree": 53,
            "disagree": 43,
            "discuss": 65, 
        },
        "discuss": {
            "agree": 214,
            "disagree": 33,
            "discuss": 1726, 
        }
    }
    gold, preds = consruct_lists(bi_att_bi_condition)
    print "Conditional bi_att Micro F1 Score: {}".format(f1_score(gold, preds, labels=LABELS, average='micro'))

    # Deep Learning Bidirectional Attention Bidirectional conditional encoding lstm
    deep_learning_bi_att_bi_condition = {
        "agree": {
            "agree": 424,
            "disagree":  0,
            "discuss": 231,        
        },
        "disagree": {
            "agree": 79,
            "disagree":  0,
            "discuss": 82, 
        },
        "discuss": {
            "agree": 274,
            "disagree":  0,
            "discuss": 1699, 
        }
    }
    gold, preds = consruct_lists(deep_learning_bi_att_bi_condition)
    print "Conditional deep_learning_bi_att_bi_condition Micro F1 Score: {}".format(f1_score(gold, preds, labels=LABELS, average='micro'))

