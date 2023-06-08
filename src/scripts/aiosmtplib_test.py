# -*- coding: utf-8 -*-
# @Time: 2023/6/8 9:44
import asyncio
import logging
import aiosmtplib
from email.header import Header
from email.mime.text import MIMEText
from email.utils import parseaddr, formataddr


class EmailConfig:
    FROM_ADDRESS = "test@163.com"
    TO_ADDRESS = "test@163.com"
    SMTP_SERVER = 'smtp.163.com'
    SMTP_AUTH_CODE = 'test_code'
    LINK = 'www.baidu.com'
    PORT = 465


def _format_address(s):
    name, address = parseaddr(s)
    return formataddr((Header(name, 'utf-8').encode(), address))


async def send_email(to_address):
    title = '<html><body><h3>亲爱的<a data-auto-link="1" href="mailto:%s" target="_blank">%s</a>, 您好: </h3>' % (
    to_address, to_address)
    reset = "<div style = 'padding-left:55px;padding-right:55px;font-family:'微软雅黑','黑体',arial;font-size:14px;'>重置密码</div>"
    body = '<p>请点击如下连接进行重置密码 <a href="%s">%s</a></p>' % (EmailConfig.LINK, reset)
    tail = '若您并非 Awesome 用户, 请忽略当前邮件</body></html>'
    html = title + body + tail

    msg = MIMEText(html, 'html', 'utf-8')
    msg['From'] = _format_address('这是邮件标题 <%s>' % EmailConfig.FROM_ADDRESS)
    msg['To'] = _format_address('亲爱的用户 <%s>' % EmailConfig.TO_ADDRESS)
    msg['Subject'] = Header('重置密码', 'utf-8').encode()

    try:
        print(">>>>>>>>>>>>> 开始发送邮件 <<<<<<<<<<<<<<<<<")
        async with aiosmtplib.SMTP(hostname=EmailConfig.SMTP_SERVER, port=EmailConfig.PORT, use_tls=True) as smtp:
            await smtp.login(EmailConfig.FROM_ADDRESS, EmailConfig.SMTP_AUTH_CODE)
            print(">>>>>>>>>>>>> 登陆成功 <<<<<<<<<<<<<<<<")
            await smtp.send_message(msg)
            print(">>>>>>>>>>>>> 发送成功 <<<<<<<<<<<<<<<<")
    # except aiosmtplib.SMTPException as e:
    except Exception as e:
        logging.error(e)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(send_email(EmailConfig.TO_ADDRESS))
