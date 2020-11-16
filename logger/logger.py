#-*- coding: utf-8 -*-

import os
import logging
import re
from logging.handlers import TimedRotatingFileHandler
from logger.MyTimedRotatingFileHandler import MyTimedRotatingFileHandler


# Loging Level
##########################################################################################################
# DEBUG < INFO < WARNING < ERROR < CRITICAL
# DEBUG	���� ����. ���� ������ ������ ���� ���
# INFO	������ �۵��ϴ� ���� ���� Ȯ��
# WARNING	����ġ ���� ���� �߻��߰ų� ����� �̷��� �߻��� ����(���� ��� ����ũ ���� ������)�� ���� ǥ��.
#        ����Ʈ����� ������ ������ �۵�
# ERROR	���� �ɰ��� ������ ����, ����Ʈ��� �Ϻ� ����� �������� ����
# CRITICAL	�ɰ��� ����. ���α׷� ��ü�� ��� ������� ���� �� ������ ��Ÿ��
##########################################################################################################


# ���ϸ� & ���丮
log_dir = './logs'
filename = log_dir + '/logs.log'

# ������ ������ ����
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
#�⺻ �ΰ�
logger = logging.getLogger(__name__)
# 2020-09-16 13:46:42,443 [    INFO] test logging �̷������� ������ ����
formatter = logging.Formatter(u'%(asctime)s [%(levelname)s|%(filename)s:%(lineno)s] %(message)s')
# setLevel�� �̿��Ͽ� Ư�� Login Level�� ��쿡�� ��ϵǵ��� �Ҽ��ִ�.
# ���⼭�� DEBUG�̹Ƿ� ����츦 ���
logger.setLevel(logging.INFO)


#fileHandler = TimedRotatingFileHandler(filename=log_dir + '/test.log', when='midnight', interval=1, encoding='utf-8')
fileHandler = MyTimedRotatingFileHandler(filename)
fileHandler.setFormatter(formatter)
fileHandler.suffix = '%Y%m%d'
fileHandler.extMatch = re.compile(r"^\d{8}$")
# fileHandler.setLevel(logging.CRITICAL)
logger.addHandler(fileHandler)

# Console�� log �����
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)
streamHandler.setLevel(logging.DEBUG)
logger.addHandler(streamHandler)

