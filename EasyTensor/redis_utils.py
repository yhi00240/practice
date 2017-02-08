# -*- coding:utf-8 -*-
from django_redis import get_redis_connection

def get_redis():
    return get_redis_connection('default')

class RedisManager():

    @staticmethod
    def get_message(practice_name):
        r = get_redis()
        return r.get(practice_name)

    @staticmethod
    def set_message(practice_name, message):
        r = get_redis()
        r.set(practice_name, message)

    @staticmethod
    def delete(practice_name):
        r = get_redis()
        r.delete(practice_name)
