#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
"""
CS 224N 2016-2017 
__init__.py: Defines models module.
Sahil Chopra <schopra8@cs.stanford.edu>
Saachi Jain <saachi@cs.stanford.edu>
John Sholar <jmsholar@cs.stanford.edu>
"""

from enum import Enum

class Labels(Enum):
    AGREE = 0
    DISAGREE = 1
    DISCUSS = 2
    UNRELATED = 3

LABEL_MAPPING = {
    'agree': 0,
    'disagree': 1,
    'dicuss': 2,
    'unrelated': 3,
}