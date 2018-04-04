# -*- encoding:utf-8 -*-
from Jconfig import *

init_dir()
if pd_config.min_example < 1:
    pd_config.min_example = 1
flog = open(pd_config.loggerPath + "cluster.log", 'a', encoding='utf8')


def mysql_cursor():
    db = pymysql.connect(
        host=pd_config.host,
        port=pd_config.port,
        user=pd_config.user,
        passwd=pd_config.password,
        db=pd_config.db,
        charset='utf8')
    return db


def get_data(day, filename, break_time):
    import jieba.analyse
    db = mysql_cursor()
    cur = db.cursor()
    while True:
        now = datetime.datetime.now()
        if now > break_time:
            flog.write(now.strftime("%Y-%m-%d %H:%M:%S ") + 'break\n')
            return False
        # print("SELECT COUNT(*) FROM %s.%s WHERE TASKID='END';COMMIT;" % (pd_config.db, pd_config.raw_table))
        cur.execute("COMMIT;")
        cur.execute("SELECT COUNT(*) FROM %s.%s WHERE TASKID='END';" % (pd_config.db, pd_config.raw_table))
        result = cur.fetchall()
        cur.execute("SELECT * FROM %s.%s limit 10;" % (pd_config.db, pd_config.raw_table))
        result1 = cur.fetchall()
        for vv in result1:
            print(vv)
        #print(result[0])
        cur.execute("COMMIT;")
        if result[0][0] != 0:
            flog.write(now.strftime("%Y-%m-%d %H:%M:%S ") + 'get data\n')
            flog.flush()
            break
        else:
            flog.write(now.strftime("%Y-%m-%d %H:%M:%S ") + 'no data\n')
            flog.flush()
            time.sleep(2)

    data = pd.read_sql("SELECT TASKID, STREETCODE, COMMUNITYCODE, GRIDCODE,\
                    INFOBCCODE, INFOSCCODE, INFOTYPEID, DESCRIPTION, COORDX,\
                    COORDY, DISCOVERTIME, ADDRESS, SERVICETYPE, INFOZCCODE, EXECUTEDEPTCODE \
                     FROM %s.%s WHERE TASKID !='END'" % (pd_config.db, pd_config.raw_table), con=db)
    # 清楚描述字符串中的回车符号
    flag_temp = 0
    for meta in data['DESCRIPTION']:
        data['DESCRIPTION'][flag_temp] = meta.replace('\n', '')
        flag_temp += 1
    KEYWORDS = []
    allow_pos =('nr', 'v','an','i','n','ns','nt','nz','vn','vd','un')

    for DESCRIPTION in data['DESCRIPTION']:
        content = re.sub("[(（][^)）]*[)）]|[\d.]+", '', DESCRIPTION)
        if len(content) < 10:
            max_num = 3
        elif len(content) < 30:
            max_num = 5
        elif len(content) < 50:
            max_num = 8
        else:
            max_num = 10
        tags = jieba.analyse.extract_tags(content, topK=max_num + 4, withWeight=False, allowPOS=allow_pos)
        words = []
        end_num = max_num
        for idx, v in enumerate(tags):
            if v in ['称该','要求','答复','求助','已有','来电','小区','人称','现象', '地址', '工单','诉求','存在', '处理结果', '回复', '验收', '关门', '山路', '松林', '大楼', '市民','请速', '翻开', '反映', '爱心', '处理', '该处', '望风', '信访办', '核实']:
                end_num += 1
                continue
            if idx == end_num:
                break
            words.append(v)
        # print("---",words)
        if len(words) == 0:
            WORDS = content.replace(" ", "")
        else:
            WORDS = " ".join(words)
        KEYWORDS.append(WORDS)
    data.insert(15, 'KEYWORDS', KEYWORDS)
    data['DAYTIME'] = day
    db.close()
    if os.path.exists(filename):
        data_before = pd.read_csv(filename, sep=u'\t', encoding='utf8')
        data_combine = data_before.append(data,ignore_index=True)
        data_combine = data_combine.drop_duplicates(['TASKID'])
        data_combine.to_csv(filename, sep=u'\t', mode='w', index=False, encoding='utf8')
    else:
        data.to_csv(filename, sep=u'\t', index=False, encoding='utf8')
    return True


def clear_mysql_data():
    db = mysql_cursor()
    cur = db.cursor()
    cur.execute("DELETE FROM %s.%s WHERE 1=1;" % (pd_config.db, pd_config.raw_table))
    cur.execute("COMMIT;")
    #cur.execute("DELETE FROM %s.%s;" % (pd_config.db, pd_config.result_table))
    #cur.execute("COMMIT;")
    #cur.execute("SELECT COUNT(*) FROM %s.%s;" % (pd_config.db, pd_config.result_table))
    #result = cur.fetchall()
    db.close()


def upload_data(resultname, filename):
    import pandas as pd
    from sqlalchemy import create_engine
    # 每次写入前清空结果表
    db = mysql_cursor()
    cur = db.cursor()
    cur.execute("DELETE FROM %s.%s;" % (pd_config.db, pd_config.result_table))
    cur.execute("COMMIT;")
    db.close()
    data = pd.read_csv(resultname)
    data = data.drop(['idx'], axis=1)
    engine_str = 'mysql+pymysql://'+pd_config.user + ':' + pd_config.password + '@' + pd_config.host + ':' + str(pd_config.port) + '/' + pd_config.db + '?charset=utf8'
    info = pd.read_csv(filename, sep=u'\t')
    res = pd.merge(data, info, how='left', on='TASKID')
    res_ = res.loc[:, ['TASKID', 'LABEL', 'INFOBCCODE', 'INFOSCCODE', 'STREETCODE', 'DISCOVERTIME', 'ADDRESS', 'DESCRIPTION', 'KEYWORDS', 'DAYTIME']]
    # con = create_engine('mysql+pymysql://user_temp_wg:fgytehjds3456hgnh@202.120.58.119:20001/db_temp_wg?charset=utf8')
    print(engine_str)
    con = create_engine(engine_str, encoding='utf-8')
    res_.to_sql(pd_config.result_table, con, if_exists='append', index=False)


def statistic_description_byevent(filename, save=True):
    '''
        Generate meta data for clustering.
    '''
    import json
    import math
    import fileinput
    import numpy as np
    from collections import defaultdict
    from sklearn.externals import joblib
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer

    file = open(pd_config.dataPath + 'stopwords.txt', 'r', encoding='utf8')
    stopwords = []
    for line in file.readlines():
        stopwords.append(line.strip())
    stopwords = set(stopwords)
    filter_stopwords = lambda l: filter(lambda x: x not in stopwords, l)
    case_dict = defaultdict(list)

    fileinput = open(filename, 'r', encoding='utf8')
    TASKID_LIST = []
    for line in fileinput.readlines()[1:]:
        try:
            TASKID, STREETCODE, COMMUNITYCODE, GRIDCODE, INFOBCCODE, INFOSCCODE, INFOTYPEID, \
            DESCRIPTION, COORDX, COORDY, DISCOVERTIME, ADDRESS, SERVICETYPE, INFOZCCODE, \
            EXECUTEDEPTCODE,KEYWORDS,DAYTIME = map(lambda x: x.strip(), line.split(u'\t'))
            # 不使用INFOBCCODE, INFOSCCODE, INFOTYPEID字段标注事件性质
            INFOBCCODE = INFOSCCODE = INFOTYPEID = '1'
            if TASKID not in TASKID_LIST:
                TASKID_LIST.append(TASKID)
            else:
                continue
            # 把坐标为0的值赋值为1，即变为非零
            if ((COORDX == '0.0') and (COORDY == '0.0')) or ((COORDX == '0') and (COORDY == '0')):
                COORDX = '1'
                COORDY = '1'
            if COORDX and COORDY and float(COORDX) and float(COORDY) and DISCOVERTIME and DESCRIPTION:
                DESCRIPTION = re.sub("[(（][^)）]*[)）]|[\d.]+", '', DESCRIPTION)
                segments = filter_stopwords(jieba.cut(DESCRIPTION))
                TEXT = re.sub(u'\s+', ' ',
                              re.sub(u'[^0-9a-zA-Z\u4e00-\u9fa5]', ' ', ' '.join(segments))).strip().lower()
                timeslot = int(time.strftime('%j', time.strptime(DISCOVERTIME, '%Y/%m/%d %H:%M'))) + \
                           (int(time.strftime('%Y', time.strptime(DISCOVERTIME, '%Y/%m/%d %H:%M'))) - 2015) * 365
                case_dict[u'{0}\t{1}\t{2}'.format(INFOTYPEID, INFOBCCODE, INFOSCCODE)].append(
                    [TASKID, timeslot, float(COORDX), float(COORDY), TEXT])
        except:

            continue
    fileinput.close()

    with open(pd_config.path + 'models/statistic_description_byevent.json', 'w') as outfile:
        outfile.write(json.dumps(case_dict))
    texts = []
    for INFONAME, TUPLE in case_dict.items():
        texts.extend(map(lambda x: x[4], TUPLE))
    # TBD: use embeddings rather than tfidf to compute text similarities
    # print(texts)
    transformer, vectorizer = TfidfTransformer(), CountVectorizer(min_df=0.00001, max_df=0.5)

    X = transformer.fit_transform(vectorizer.fit_transform(texts)).toarray()

    if save:
        joblib.dump(transformer, pd_config.path + 'models/transformer.model')
        joblib.dump(vectorizer, pd_config.path + 'models/vectorizer.model')
    else:
        return transformer, vectorizer


def spatio_temporal_semantic_clustering(resultname, e, filename):
    '''
        Clustering by spatio, temporal and semantic meta data.
        TBD: consider cross category clustering (since different categories can have high mutual information)
    '''
    import json
    from sklearn.externals import joblib
    from sklearn.metrics import normalized_mutual_info_score

    case_dict = json.loads(open(pd_config.path + 'models/statistic_description_byevent.json').read())
    try:
        transformer = joblib.load(pd_config.path + 'models/transformer.model')
        vectorizer = joblib.load(pd_config.path + 'models/vectorizer.model')
    except:
        transformer, vectorizer = statistic_description_byevent(filename, save=False)

    import math
    import pandas as pd
    from sklearn.cluster import AgglomerativeClustering, DBSCAN
    import numpy as np

    def getkey(item):
        return item[1]

    def compute_similarity_time(t1, t2, thres=30, delta=2):
        if abs(t1 - t2) >= thres:
            return 0
        else:
            return math.exp(-1. * abs(t1 - t2) / delta)

    def compute_similarity_space(coor1, coor2, thres=200, delta=300):
        distance = sum((coor1[i] - coor2[i]) ** 2 for i in range(2)) ** 0.5
        return math.exp(-1. * distance / delta)
        #if distance >= thres:
            #return 0
        #else:
            #return math.exp(-1. * distance / delta)

    def compute_similarity_texts(vec1, vec2, min_value=0.1, threshold=0.25):
        # TBD: should consider the text length impact
        cosine_similarity = (vec1 * vec2).sum() / ((vec1 ** 2).sum() * (vec2 ** 2).sum()) ** 0.5
        if math.isnan(cosine_similarity):
            res = 0
        else:
            res = cosine_similarity
        if res < threshold:
            return 0
        return res

    with open(resultname, 'w', encoding='utf8') as outfile:
        print(u'total category:\t', len(case_dict))
        result = pd.DataFrame()
        EVENTID = e
        no = []
        outfile.write(u'{0},{1},{2}\n'.format('TASKID', 'LABEL', 'idx'))
        #fww = open('temp_result.txt', 'w', encoding='utf8')
        for idx, (INFONAME, LIST) in enumerate(sorted(case_dict.items())):
            # print(u'{0}, {1}'.format(idx, len(LIST)))
            '''all_list = sorted(LIST, key=getkey)
            for t in range(0, len(all_list), 10000):
                if len(all_list[t:]) < 10000:
                    LIST = all_list[t:]
                else:
                    LIST = all_list[t:t + 10000]'''
            similarity_matrix = np.zeros((len(LIST), len(LIST)))
            taskids = list(map(lambda x: x[0], LIST))
            texts = map(lambda x: x[4], LIST)
            vecs = transformer.transform(vectorizer.transform(texts)).toarray()
            for i in range(len(LIST)):
                for j in range(i, len(LIST)):
                    similarity_time = compute_similarity_time(LIST[i][1], LIST[j][1])
                    similarity_space = 0 if not similarity_time else compute_similarity_space(LIST[i][2:4],
                                                                                              LIST[j][2:4])
                    similarity_texts = 0 if not similarity_time or not similarity_space else compute_similarity_texts(
                        vecs[i], vecs[j])

                    similarity_matrix[i, j] = similarity_matrix[j, i] = (similarity_time * similarity_space * similarity_texts) ** (1. / 3)
                    #fww.write(str(similarity_time)+'\t'+str(similarity_space)+'\t'+str(similarity_texts)+'\t'+str(similarity_matrix[i, j])+'\n')
                    # similarity_matrix[i, j] = similarity_matrix[j, i] = (similarity_time * similarity_space) ** (1. / 3)
            matrix_distance = 1. / (similarity_matrix + 0.1 ** 5)
            #matrix_distance = similarity_matrix

            clustering = DBSCAN(eps=2, min_samples=pd_config.min_example, metric='precomputed', n_jobs=-1)
            clustering.fit(matrix_distance)

            tmp = []
            use = []
            clusters = list(clustering.labels_)
            for i in range(len(LIST)):
                # labels从0开始
                if clusters[i] == - 1:
                    print('出现聚类异常数据')
                    clusters[i] = len(clusters)
                if clusters.count(clusters[i]) < pd_config.min_example:
                    print('出现clusters.count(clusters[i]) < pd_config.min_example')
                    continue
                use.append(i)
                if clusters[i] not in tmp:
                    tmp.append(clusters[i])
                #if clusters[i] == - 1 or clusters.count(clusters[i]) < pd_config.min_example:
                    #print('出现聚类异常数据，会导致结果变少')
                    #continue
                #else:
                    #use.append(i)
                    #if clusters[i] not in tmp:
                        #tmp.append(clusters[i])

            for j in use:
                outfile.write(u'{0},{1},{2}\n'.format(taskids[j], EVENTID + tmp.index(clusters[j]), idx))

            EVENTID += len(tmp)
                #print(u'{0},{1}'.format(len(tmp), t))
        outfile.close()
        #fww.close()
    return EVENTID


def main(day, break_time):
    import time
    filename = pd_config.path + 'data_' + day + '.csv'
    resultname = pd_config.path + 'result_' + day + '.csv'
    e = 0
    if not get_data(day, filename, break_time):
        return
    statistic_description_byevent(filename, save=True)
    spatio_temporal_semantic_clustering(resultname, e, filename)
    hour = time.strftime('%H',time.localtime(time.time()))
    print(hour)
    if hour == '00':
        print('...删除T_READ2USE...')
        clear_mysql_data()
    upload_data(resultname, filename)


if __name__ == "__main__":
    import time
    import datetime
    # 时间减1个小时，计算前一天的情况。每个小时定时从10分钟开始跑
    now = datetime.datetime.now() - datetime.timedelta(hours=1)
    break_time = datetime.datetime.now() + datetime.timedelta(minutes=50)
    day = now.strftime("%Y-%m-%d")
    print(day)
    # 删除前天产生的.csv文件,now2='2018-03-31'
    now2 = datetime.datetime.now()
    delta = datetime.timedelta(days=-1)
    now2 = now2 + delta
    now2 = now2.strftime('%Y-%m-%d')
    filename1 = 'result_' + now2 + '.csv'
    filename2 = 'data_' + now2 + '.csv'
    if os.path.exists(filename1):
        os.remove(filename1)
    if os.path.exists(filename2):
        os.remove(filename2)
    main(day, break_time)
