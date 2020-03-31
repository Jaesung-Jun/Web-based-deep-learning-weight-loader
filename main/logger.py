import time
from datetime import datetime
import os

LOG_PATH = "./log"

def createLogDirectory():
    if not os.path.exists(LOG_PATH):
        print("Created Log Directory")
        os.makedirs(LOG_PATH)

def createLogFile(username):
    if not os.path.exists(LOG_PATH + "/log_{}.log".format(username)):
        print("{} : Created log File".format(username))
        f = open(LOG_PATH + "/log_{}.log".format(username), 'w')
        f.close()

def log(username, string, important=False):

    createLogDirectory()
    createLogFile(username)
    
    now = datetime.now()
    current_time = "[{0}]".format(now.isoformat(sep=" "))

    if important:
        ipt = "★ "
    else:
        ipt = ""
    
    log_message = "{0}{1} {2}\n".format(ipt, current_time, string)

    f = open(LOG_PATH + "/log_{}.log".format(username), 'a')
    f.write(log_message)
    f.close()