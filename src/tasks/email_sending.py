# -*- coding: utf-8 -*-
# @Time: 2023/6/7 10:20
import os
import smtplib
import asyncio
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from config.redis_config import REDIS_CONN


class SendEmail:

    EMAIL_HOST_USER = "shawnhuang90s@163.com"
    EMAIL_AUTH_PWD = "JODBLDPXHFGCAQKP"
    EMAIL_HOST = "smtp.163.com"
    EMAIL_PORT = 465

    def __init__(self, user=EMAIL_HOST_USER, password=EMAIL_AUTH_PWD, host=EMAIL_HOST, port=EMAIL_PORT,
                 doc=None, content=None, tag=None, cc_list=None, to_list=None):
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.doc = doc
        self.content = content
        self.tag = tag
        self.cc_list = cc_list
        self.to_list = to_list

    def structure_email_content(self):
        attach = MIMEMultipart()
        if self.user:
            attach['From'] = self.user
        if self.tag:
            attach['Subject'] = self.tag
        if self.content:
            email_content = MIMEText(
                self.content,
                'plain',
                'utf-8'
            )
            attach.attach(email_content)
        if self.to_list:
            if isinstance(self.to_list, list):
                attach["To"] = ";".join(self.to_list)
            elif isinstance(self.to_list, str):
                attach["To"] = self.to_list
        if self.doc:
            name = os.path.basename(self.doc)
            f = open(self.doc, 'rb')
            doc = MIMEText(f.read(), 'base64', 'utf-8')
            doc['Content-Type'] = 'application/octet-stream'
            doc['Content-Disposition'] = 'attachment; filename="' + name + '"'
            attach.attach(doc)
            f.close()

        return attach.as_string()

    async def send(self):
        server = None
        try:
            # server = smtplib.SMTP(self.host, self.port)
            server = smtplib.SMTP_SSL(self.host, self.port)
            server.login(self.user, self.password)
            print(f'Login successfully')
            server.sendmail(self.user, self.to_list, self.structure_email_content())
            server.close()
            print(f'Send to {self.to_list} successfully')
        except Exception as e:
            print(f'Send server failed: {e}')
            if server is not None:
                server.close()


class TimeTask:

    REDIS_NAME = "email_users"
    EMAIL_TAG = "This is a test..."
    EMAIL_USER = "shawnhuang90s@163.com"

    async def run(self):
        print(">>>>>>>>>> Run send email <<<<<<<<<<")
        await REDIS_CONN.hset(name=self.REDIS_NAME, mapping={"test_name": "test_pwd"})
        users_dict = await REDIS_CONN.hgetall(name=self.REDIS_NAME)
        if len(users_dict) == 0:
            print("No user should be send.")
            return
        for username, password in users_dict.items():
            content = f"Hello, {username}, this is your password: {password}"
            email_obj = SendEmail(content=content, tag=self.EMAIL_TAG, to_list=self.EMAIL_USER)
            await email_obj.send()
            await REDIS_CONN.hdel(self.REDIS_NAME, username)
            await asyncio.sleep(10)
        await REDIS_CONN.close()


if __name__ == '__main__':
    # loop = asyncio.get_event_loop()
    # loop.create_task(TimeTask().run())
    # loop.close()
    asyncio.run(TimeTask().run())
