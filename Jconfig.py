# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import os
import pymysql
import pandas as pd
import re
import os
import jieba
import time
import datetime


'''
 * 
 * <p>Title: </p>
 * <p>Description: 基本的配置文件，包含文件名，配置文件路径等 </p>
 * <p>Copyright (c) 2017 </p>
 * <p>Company: seassoon </p>
 * @Author: wjy
 * 25 Dec 2017: 25 Oct 2017 11:13 AM
 * @Version 1.0
'''


class AttrDict(dict):
    """ Dict that can get attribute by dot """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


pd_config = AttrDict()
# 数据库配置
pd_config.host = 'localhost'
pd_config.port = 3306
pd_config.user = 'root'
pd_config.password = 'SQLsql123'
pd_config.db = 'db_wg'
# 每小时定时任务表
pd_config.raw_table = 'T_READY2USE'
pd_config.result_table = 'T_INFO_RESULT_TEST'

# 每月定时任务表：
pd_config.keywords_inf_table = 'T_TASKINFO'
pd_config.keywords_result_table = 'T_KEYWORDS_RESULT'
# 脚本路径
pd_config.path = os.path.abspath("./") + "/"
if not pd_config.path.endswith('/'):
    pd_config.path += '/'

pd_config.dataPath = pd_config.path + 'data'
if not pd_config.dataPath.endswith('/'):
    pd_config.dataPath += '/'

pd_config.loggerPath = pd_config.path + 'logger'
if not pd_config.loggerPath.endswith('/'):
    pd_config.loggerPath += '/'
print(pd_config)
# 聚类最小单位 和 聚类结果文件夹
pd_config.min_example = 1
pd_config.bit_event_dir = pd_config.path + "big_event_input/"
pd_config.result_dir = pd_config.path + "big_event_cluster/"

# 重大案件表
pd_config.big_event_table = "leader_media_discover"
pd_config.history_table = 'T_TASKINFO_BAK'
pd_config.on_table = 'T_TASKINFO'

columns = ['TASKID','STREETCODE','COMMUNITYCODE','GRIDCODE','INFOBCCODE','INFOSCCODE','INFOTYPEID','COORDX','COORDY','DISCOVERTIME','SERVICETYPE','INFOZCCODE','EXECUTEDEPTCODE','INFOSOURCENAME','ADDRESS','DESCRIPTION' ]


def init_dir():
    if not os.path.exists(pd_config.path + "cluster.log"):
        os.system("touch " + pd_config.loggerPath + "cluster.log")