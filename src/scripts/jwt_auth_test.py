# -*- coding: utf-8 -*-
# @Time: 2023/6/8 9:52
import time
import jwt
from fastapi import HTTPException
from passlib.context import CryptContext


class AuthHandler:

    pwd_context = CryptContext(schemes=['bcrypt'], deprecated='auto')
    secret = 'custom_secret_value'
    algorithm = 'HS256'
    expiration = int(time.time() * 1000) + 30 * 1000

    def get_password_hash(self, password):
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password, hashed_password):
        return self.pwd_context.verify(plain_password, hashed_password)

    def encode_token(self):
        payload = {
            'exp': self.expiration,
            'data': '{}',
        }
        jwt_token = str(jwt.encode(payload, self.secret, self.algorithm))
        return jwt_token

    def decode_token(self, jwt_token):
        try:
            payload = jwt.decode(jwt_token, self.secret, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail='Signature has expired')
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail='Invalid token')


if __name__ == '__main__':
    auth_obj = AuthHandler()
    jwt_token = auth_obj.encode_token()
    print(jwt_token)
    payload = auth_obj.decode_token(jwt_token)
    print(payload)
