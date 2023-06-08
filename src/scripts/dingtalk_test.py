# -*- coding: utf-8 -*-
# @Time: 2023/6/8 10:24
"""
参考资料：
    https://open.dingtalk.com/document/robots/robot-overview
    https://blog.csdn.net/u010751000/article/details/121313045
"""
import time
import json
import hmac
import base64
import hashlib
import requests
from urllib.parse import quote_plus


class Messager:

    def __init__(self, token, url, secret):
        self.token = token
        self.url = url
        self.secret = secret
        self.timestamp = str(round(time.time() * 1000))
        self.secret_enc = self.secret.encode('utf-8')
        self.string_to_sign = '{}\n{}'.format(self.timestamp, secret)
        self.string_to_sign_enc = self.string_to_sign.encode('utf-8')
        self.hmac_code = hmac.new(self.secret_enc, self.string_to_sign_enc, digestmod=hashlib.sha256).digest()
        self.sign = quote_plus(base64.b64encode(self.hmac_code))
        self.headers = {'Content-Type': 'application/json'}
        self.params = {'access_token': self.token, 'sign': self.sign, 'timestamp': self.timestamp}

    def send_simple_text(self, content):
        data = {
            'msgtype': 'text',
            'at': {
                'atMobiles': [15018267752]  # The person who wants @ can use the bound phone number
            },
            'text': {'content': content}
        }
        return requests.post(url, data=json.dumps(data), params=self.params, headers=self.headers)

    def send_markdown_text(self, title, markdown_content):
        data = {
            'msgtype': 'markdown',
            'markdown': {
                'title': title,
                'text': markdown_content
            }
        }
        return requests.post(url, data=json.dumps(data), params=self.params, headers=self.headers)


if __name__ == '__main__':
    token = "dingtalk_token"
    url = "https://oapi.dingtalk.com/robot/send"
    secret = 'dingtalk_secret'
    msg = Messager(token=token, url=url, secret=secret)
    r = msg.send_simple_text('Send msg test...')
    print(r)
    print(r.text)

    # markdown_content = '\n'.join(open('C:\\Users\abc\Desktop\test.md', encoding='utf-8').readlines())
    # r = msg.send_markdown_text(title='Test markdown content', markdown_content=markdown_content)
    # print(r.text)
