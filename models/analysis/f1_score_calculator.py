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

def results():
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

    # Bimpm
    bimpm = {
        "agree": {
            "agree": 421,
            "disagree":  13,
            "discuss": 211,        
        },
        "disagree": {
            "agree": 56,
            "disagree":  17,
            "discuss": 88, 
        },
        "discuss": {
            "agree": 296,
            "disagree":  0,
            "discuss": 1669, 
        }
    }
    gold, preds = consruct_lists(bimpm)
    print "Conditional bimpm Micro F1 Score: {}".format(f1_score(gold, preds, labels=LABELS, average='micro'))


def hyperparameters():
    output_1 = {
        "agree": {
            "agree": 365,
            "disagree":  17,
            "discuss": 164,        
        },
        "disagree": {
            "agree": 86,
            "disagree": 16,
            "discuss": 55, 
        },
        "discuss": {
            "agree": 148,
            "disagree": 2,
            "discuss": 1220, 
        }
    }
    gold, preds = consruct_lists(output_1)
    print "Conditional output_1 Micro F1 Score: {}".format(f1_score(gold, preds, labels=LABELS, average='micro'))

    output_2 = {
        "agree": {
            "agree": 546,
            "disagree":  0,
            "discuss": 0,        
        },
        "disagree": {
            "agree": 157,
            "disagree": 0,
            "discuss": 0, 
        },
        "discuss": {
            "agree": 1370,
            "disagree": 0,
            "discuss": 0, 
        }
    }
    gold, preds = consruct_lists(output_2)
    print "Conditional output_2 Micro F1 Score: {}".format(f1_score(gold, preds, labels=LABELS, average='micro'))

    output_3 = {
        "agree": {
            "agree": 370,
            "disagree":  0,
            "discuss": 176,        
        },
        "disagree": {
            "agree": 113,
            "disagree": 0,
            "discuss": 44, 
        },
        "discuss": {
            "agree": 124,
            "disagree": 0,
            "discuss": 1246, 
        }
    }
    gold, preds = consruct_lists(output_3)
    print "Conditional output_3 Micro F1 Score: {}".format(f1_score(gold, preds, labels=LABELS, average='micro'))

    output_4 = {
        "agree": {
            "agree": 388,
            "disagree":  0,
            "discuss": 158,        
        },
        "disagree": {
            "agree": 116,
            "disagree": 0,
            "discuss": 41, 
        },
        "discuss": {
            "agree": 173,
            "disagree": 0,
            "discuss": 1197, 
        }
    }
    gold, preds = consruct_lists(output_4)
    print "Conditional output_4 Micro F1 Score: {}".format(f1_score(gold, preds, labels=LABELS, average='micro'))

    output_5 = {
        "agree": {
            "agree": 546,
            "disagree":  0,
            "discuss": 0,        
        },
        "disagree": {
            "agree": 157,
            "disagree": 0,
            "discuss": 0, 
        },
        "discuss": {
            "agree": 1370,
            "disagree": 0,
            "discuss": 0, 
        }
    }
    gold, preds = consruct_lists(output_5)
    print "Conditional output_5 Micro F1 Score: {}".format(f1_score(gold, preds, labels=LABELS, average='micro'))

    output_6 = {
        "agree": {
            "agree": 347,
            "disagree":  0,
            "discuss": 199,        
        },
        "disagree": {
            "agree": 93,
            "disagree": 0,
            "discuss": 64, 
        },
        "discuss": {
            "agree": 222,
            "disagree": 0,
            "discuss": 1148, 
        }
    }
    gold, preds = consruct_lists(output_6)
    print "Conditional output_6 Micro F1 Score: {}".format(f1_score(gold, preds, labels=LABELS, average='micro'))

    output_7 = {
        "agree": {
            "agree": 343,
            "disagree":  26,
            "discuss": 177,        
        },
        "disagree": {
            "agree": 87,
            "disagree": 37,
            "discuss": 33, 
        },
        "discuss": {
            "agree": 172,
            "disagree": 0,
            "discuss": 1190, 
        }
    }
    gold, preds = consruct_lists(output_7)
    print "Conditional output_7 Micro F1 Score: {}".format(f1_score(gold, preds, labels=LABELS, average='micro'))

    output_8 = {
        "agree": {
            "agree": 326,
            "disagree":  43,
            "discuss": 177,        
        },
        "disagree": {
            "agree": 73,
            "disagree": 38,
            "discuss": 46, 
        },
        "discuss": {
            "agree": 144,
            "disagree": 17,
            "discuss": 1208, 
        }
    }
    gold, preds = consruct_lists(output_8)
    print "Conditional output_8 Micro F1 Score: {}".format(f1_score(gold, preds, labels=LABELS, average='micro'))

    output_9 = {
        "agree": {
            "agree": 347,
            "disagree":  61,
            "discuss": 138,        
        },
        "disagree": {
            "agree": 73,
            "disagree": 61,
            "discuss": 23, 
        },
        "discuss": {
            "agree": 154,
            "disagree": 41,
            "discuss": 1175, 
        }
    }
    gold, preds = consruct_lists(output_9)
    print "Conditional output_9 Micro F1 Score: {}".format(f1_score(gold, preds, labels=LABELS, average='micro'))

    output_10 = {
        "agree": {
            "agree": 273,
            "disagree":  44,
            "discuss": 229,        
        },
        "disagree": {
            "agree": 58,
            "disagree": 47,
            "discuss": 52, 
        },
        "discuss": {
            "agree": 107,
            "disagree": 14,
            "discuss": 1249, 
        }
    }
    gold, preds = consruct_lists(output_10)
    print "Conditional output_10 Micro F1 Score: {}".format(f1_score(gold, preds, labels=LABELS, average='micro'))

    output_11 = {
        "agree": {
            "agree": 316,
            "disagree":  15,
            "discuss": 215,        
        },
        "disagree": {
            "agree": 76,
            "disagree": 10,
            "discuss": 71, 
        },
        "discuss": {
            "agree": 111,
            "disagree": 8,
            "discuss": 1251, 
        }
    }
    gold, preds = consruct_lists(output_11)
    print "Conditional output_11 Micro F1 Score: {}".format(f1_score(gold, preds, labels=LABELS, average='micro'))

    output_12 = {
        "agree": {
            "agree": 343,
            "disagree":  0,
            "discuss": 203,        
        },
        "disagree": {
            "agree": 108,
            "disagree": 0,
            "discuss": 49, 
        },
        "discuss": {
            "agree": 135,
            "disagree": 0,
            "discuss": 1235, 
        }
    }
    gold, preds = consruct_lists(output_12)
    print "Conditional output_12 Micro F1 Score: {}".format(f1_score(gold, preds, labels=LABELS, average='micro'))
 
if __name__ == '__main__':
    results()

