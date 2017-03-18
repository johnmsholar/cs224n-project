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
            "agree": 12,
            "disagree":  0,
            "discuss": 643,        
        },
        "disagree": {
            "agree": 4,
            "disagree":  0,
            "discuss": 157, 
        },
        "discuss": {
            "agree": 5,
            "disagree":  0,
            "discuss": 1968, 
        }
    }
    gold, preds = consruct_lists(conditional_lstm)

    print len(gold), len(preds)
    print "GOLD:"
    print "Agree: {}".format(gold.count(0))
    print "Disagree: {}".format(gold.count(1))
    print "Discuss: {}".format(gold.count(2))

    print "preds:"    
    print "Agree: {}".format(preds.count(0))
    print "Disagree: {}".format(preds.count(1))
    print "Discuss: {}".format(preds.count(2))

    print "Conditional LSTM Micro F1 Score: {}".format(f1_score(gold, preds, labels=LABELS, average='micro'))