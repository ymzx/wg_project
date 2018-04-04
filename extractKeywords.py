# -*- coding: utf-8 -*-
# !/usr/bin/env python3
# import os
from Jconfig import *
import os
import jieba.analyse
from sqlalchemy import create_engine
import numpy as np


'''
 * 
 * <p>Title: </p>
 * <p>Description: 提取关键词语 </p>
 * <p>Copyright (c) 2017 </p>
 * <p>Company: seassoon </p>
 * @Author: wjy
 * @Date: 29 Dec 2017 5:15 PM 
 * @Version 1.0
'''


def extract_keywords(sentence):
    allow_pos = ('nr', 'v', 'an', 'i', 'n', 'ns', 'nt', 'nz', 'vn', 'vd', 'un')
    content = re.sub("[(（][^)）]*[)）]|[\d.]+", '', sentence)
    if len(content) < 10:
        max_num = 3
    elif len(content) < 40:
        max_num = 5
    elif len(content) < 70:
        max_num = 7
    else:
        max_num = 9
    tags = jieba.analyse.extract_tags(content, topK=max_num + 4, withWeight=False, allowPOS=allow_pos)
    word = []
    end_num = max_num
    for idx, v in enumerate(tags):
        if v in ['导致','部门','希望','称该','要求','答复','求助','已有','来电','小区','人称','现象', '地址', '工单','诉求','存在', '处理结果', '回复', '验收', '关门', '山路', '松林', '大楼', '市民','请速', '翻开', '反映', '爱心', '处理', '该处', '望风', '信访办', '核实']:
            end_num += 1
            continue
        if idx == end_num:
            break
        word.append(v)
    words = " ".join(word)
    return words


def mysql_cursor():
    db = pymysql.connect(
        host=pd_config.host,
        port=pd_config.port,
        user=pd_config.user,
        passwd=pd_config.password,
        db=pd_config.db,
        charset='utf8')
    return db


def get_data(filename, month, month_next):
    db = mysql_cursor()
    print("SELECT TASKID, EXECUTEDEPTNAME, DISCOVERTIME, DESCRIPTION, BACKNOTES FROM %s WHERE BACKCOUNT > 0 and DISCOVERTIME BETWEEN '%s-01 00:00:00' and '%s-01 00:00:00' ORDER BY EXECUTEDEPTCODE,DISCOVERTIME" % (pd_config.keywords_inf_table, month, month_next))
    data = pd.read_sql("SELECT TASKID, EXECUTEDEPTNAME, DISCOVERTIME, DESCRIPTION, BACKNOTES FROM %s WHERE BACKCOUNT > 0 and DISCOVERTIME BETWEEN '%s-01 00:00:00' and '%s-01 00:00:00' ORDER BY EXECUTEDEPTCODE,DISCOVERTIME" % (pd_config.keywords_inf_table, month, month_next), con=db)
    data.to_csv(filename, sep=u'\t', index=False, encoding='utf8')
    # 市级平台


def analysis_data(filename, outname):
    data = pd.read_csv(filename, sep=u'\t', encoding='utf8')
    data = data.drop_duplicates(['TASKID'])
    KEYWORD_LIST = []
    data = data.fillna({'TASKID': "", 'EXECUTEDEPTNAME': '市级平台', 'DESCRIPTION': " ", 'BACKNOTES':""})

    for DESCRIPTION in data['DESCRIPTION']:
        KEYWORD_LIST.append(extract_keywords(DESCRIPTION))
        # print(DESCRIPTION)

    # 处理为空的派遣单位。
    data.insert(5, "KEYWORDS", KEYWORD_LIST)
    data.to_csv(outname, sep=u'\t', index=False, encoding='utf8')


def update_mysql(filename):
    engine_str = 'mysql+pymysql://'+pd_config.user + ':' + pd_config.password + '@' + pd_config.host + ':' + str(pd_config.port) + '/' + pd_config.db + '?charset=utf8'
    info = pd.read_csv(filename, sep=u'\t')
    con = create_engine(engine_str, encoding='utf-8')
    info.to_sql(pd_config.keywords_result_table, con, if_exists='append', index=False)
    pass


def main(month, month_next):
    filename = pd_config.path + 'data_' + month + '.csv'
    outname = pd_config.path + 'result_' + month + '.csv'
    if not os.path.exists(filename):
        get_data(filename, month, month_next)
    analysis_data(filename, outname)
    update_mysql(outname)


def test():
    import numpy as np
    info = pd.read_csv('test.csv', sep=u'\t')
    for x in info['DESCRIPTION']:
        if np.isnan(x):
            print(x)
        else:
            print('what')


def analysis_history():
    filename = pd_config.path + "data_2017-11.csv"
    month = "1970-01"
    month_next = "2017-12"
    outname = pd_config.path + "result_2017-11.csv"
    if not os.path.exists(filename):
        get_data(filename, month, month_next)
    if not os.path.exists(outname):
        analysis_data(filename,outname)
    update_mysql(outname)


if __name__ == "__main__":
    month_time = datetime.datetime.now() - datetime.timedelta(days=27)
    next_month_time = datetime.datetime.now()
    month = month_time.strftime("%Y-%m")
    month_next = next_month_time.strftime("%Y-%m")
    # test()
    # analysis_history()
    main(month, month_next)
    pass
