# -*- coding:utf-8 -*-
import sys,os
import logging
logger_console = logging.getLogger("ner_console")
formatter = logging.Formatter('%(process)s -- %(asctime)s %(levelname)s:  %(message)s')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

logger_console.addHandler(console_handler)
logger_console.setLevel(logging.INFO)

logger_console.info("启动日志流")
