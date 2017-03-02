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
    UNRELATED = 0
    DISCUSS = 1
    AGREE = 2
    DISAGREE = 3

LABEL_MAPPING = {
    'unrelated': Labels.UNRELATED,
    'discuss': Labels.DISCUSS,
    'agree': Labels.AGREE,
    'disagree': Labels.DISAGREE,
}
