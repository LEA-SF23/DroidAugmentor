#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
from datetime import datetime

print  (sys.argv[1:])

# datetime object containing current date and time
now = datetime.now()
 
print("now =", now)

# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print ("date and time =", dt_string)

