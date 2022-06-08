#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright Â© 2020 Lucas Costa Campos <rmk236@gmail.com>
#
# Distributed under terms of the MIT license.

import datetime
import numpy as np


def print_time():
    print("\n\n\ntime and date:\n")
    print(str(datetime.datetime.now().strftime("%I-%M%p-on-%B-%d-%Y")),"\n\n\n")
    return str(datetime.datetime.now().strftime("%I-%M%p-on-%B-%d-%Y"))


#write the txt file
def write_loss_to_txt(s, file):
    file.write(s+"\n")


#read the txt file
def read_loss_from_txt(filepath):
    file = open(filepath)
    res = file.readlines()
    res = [e.split(" ") for e in res]
    file.close()
    print(res)
    return res
