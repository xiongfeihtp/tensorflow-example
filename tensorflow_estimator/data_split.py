# coding:utf-8
import pandas as pd
import time
# instance_id 样本编号
# item_id 广告商品编号
# item_category_list 广告商品的的类目列表 分割; item_property_list_0 item_property_list_1 item_property_list_2
# item_property_list 广告商品的属性列表 分割 1 2 3
# item_brand_id 广告商品的品牌编号
# item_city_id 广告商品的城市编号
# item_price_level 广告商品的价格等级
# item_sales_level 广告商品的销量等级
# item_collected_level 广告商品被收藏次数的等级
# item_pv_level 广告商品被展示次数的等级
# user_id 用户的编号
# 'user_gender_id', 用户的预测性别编号
# 'user_age_level', 用户的预测年龄等级
# 'user_occupation_id', 用户的预测职业编号
# 'user_star_level' 用户的星级编号
# context_id 上下文信息的编号
#  context_timestamp 广告商品的展示时间
# context_page_id 广告商品的展示页面编号
# predict_category_property

def time2cov(time_):
    '''
    时间是根据天数推移，所以日期为脱敏，但是时间本身不脱敏
    :param time_:
    :return:
    '''
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_))


def pre_process(data):
    '''
    :param data:
    :return:
    '''
    # python预处理，完成后，释放一下内存
    print('预处理')
    print('item_category_list_ing')
    # item_category_list 取三个占位符
    for i in range(3):
        data['category_%d' % (i)] = data['item_category_list'].apply(
            lambda x: x.split(";")[i] if len(x.split(";")) > i else " "
        )

    del data['item_category_list']

    # item_property_list 取三个占位符
    print('item_property_list_ing')
    for i in range(3):
        data['property_%d' % (i)] = data['item_property_list'].apply(
            lambda x: x.split(";")[i] if len(x.split(";")) > i else " "
        )

    del data['item_property_list']
    # context_timestamp 转换为日期和小时
    print('context_timestamp_ing')
    data['context_timestamp'] = data['context_timestamp'].apply(time2cov)

    print('time')
    # 时间
    data['context_timestamp_tmp'] = pd.to_datetime(data['context_timestamp'])
    # 日期
    data['week'] = data['context_timestamp_tmp'].dt.weekday
    # 天数
    data['hour'] = data['context_timestamp_tmp'].dt.hour
    del data['context_timestamp_tmp']
    # 三个占位符
    print('predict_category_property_ing_0')
    for i in range(3):
        data['predict_category_%d' % (i)] = data['predict_category_property'].apply(
            lambda x: str(x.split(";")[i]).split(":")[0] if len(x.split(";")) > i else " "
        )
    del data['predict_category_property']
    # del data['predict_property_1']
    # del data['predict_property_2']
    return data


print('train')
train = pd.read_csv('round1_ijcai_18_train_20180301.txt', sep=" ")
train = pre_process(train)
# 没有做独热编码，只是将时间预处理，部分多label的特征进行了对齐，取了三个占位符
all_data = train.copy()
print('all_shape', train.shape)
print(train['context_timestamp'].max())
val = train[train['context_timestamp'] > '2018-09-23 23:59:59']

train = train[train['context_timestamp'] <= '2018-09-23 23:59:59']
# train = train[train['context_timestamp'] > '2018-09-19 23:59:59']
print(train.shape)
print(val.shape)

# 这里是增加的内容
import datetime


def get_count_feat(all_data, data, long=3):
    end_time = data['context_timestamp'].min()
    # 最小的时间，往后推三天
    begin_time = pd.to_datetime(end_time) - datetime.timedelta(days=long)
    all_data['context_timestamp'] = pd.to_datetime(all_data['context_timestamp'])
    # 取出需要统计的数据
    all_data = all_data[
        (all_data['context_timestamp'] < end_time) & (all_data['context_timestamp'] >= begin_time)
        ]
    print(end_time)
    print(begin_time)
    # 通过pandas进行计数的技巧
    print(all_data['context_timestamp'].max() - all_data['context_timestamp'].min())
    item_count = all_data.groupby(['item_id'], as_index=False).size().reset_index()
    item_count.rename(columns={0: 'item_count'}, inplace=True)

    user_count = all_data.groupby(['user_id'], as_index=False).size().reset_index()
    user_count.rename(columns={0: 'user_count'}, inplace=True)
    return user_count, item_count

# 分别取了对应数据的前三天的user个数统计
train_user_count, train_item_count = get_count_feat(all_data, train, 3)
val_user_count, val_item_count = get_count_feat(all_data, val, 3)

# 进行数据表连接，并填充默认值-1
train = pd.merge(train, train_user_count, on=['user_id'], how='left')
train = pd.merge(train, train_item_count, on=['item_id'], how='left')
train = train.fillna(-1)

val = pd.merge(val, val_user_count, on=['user_id'], how='left')
val = pd.merge(val, val_item_count, on=['item_id'], how='left')
val = val.fillna(-1)

del train['instance_id']
del train['context_timestamp']
del val['instance_id']
del val['context_timestamp']
"""
可持续化保存
"""
import os

if not os.path.exists('./train'):
    os.makedirs('./train')

train.to_csv('./train/train.csv', index=False)

if not os.path.exists('./eval'):
    os.makedirs('./eval')
val.to_csv('./eval/eval.csv', index=False)
