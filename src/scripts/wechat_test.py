# -*- coding: utf-8 -*-
# @Time: 2023/6/8 10:16
"""
企业微信机器人推送消息功能
参考资料：
    https://blog.csdn.net/wangzhneg123/article/details/117919339
    https://zhuanlan.zhihu.com/p/524515595
"""
import requests
import base64
import hashlib


def simple_run():
    url = 'wechat_reobot_url'
    header = {'Content-Type': 'application/json; charset=utf-8'}
    body = {
        'msgtype': 'text',
        'text': {
            'content': 'This is a test~'
        }
    }
    requests.post(url, json=body, headers=header)


def send_image_message():
    with open('人生大事.jpg','rb') as f:
        base64_data = base64.b64encode(f.read())
        image_data = str(base64_data,'utf-8')
    with open('人生大事.jpg','rb') as f:
        md = hashlib.md5()
        md.update(f.read())
        image_md5 = md.hexdigest()
    url = 'wechat_reobot_url'
    headers = {"Content-Type":'application/json'}
    data = {
        'msgtype':'image',
        'image':{
            'base64':image_data,
            'md5':image_md5
        }
    }
    r = requests.post(url,headers=headers,json=data)
    print(r)


if __name__ == '__main__':
    # simple_run()
    send_image_message()
