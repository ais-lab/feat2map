#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 12:09:49 2022
@author: thuan
"""

import sys

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
        
    def delink(self):
        self.log.close()

    def writeTerminalOnly(self, message):
        self.terminal.write(message)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass
    
class History(object):
    def __init__(self):
        self.num = 0
        self.cumulative = 0
        self.history = []
    def update(self, new):
        self.num += 1
        self.cumulative += new 
        self.history.append(new)
    def average(self):
        return self.cumulative/self.num
        