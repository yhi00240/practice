# -*- coding:utf-8 -*-
import fakeredis

def get_redis():
    return fakeredis.FakeStrictRedis()

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
    def get_element(step_name): #step_accuacy, step_cost의 방식으로 저장합니다.
        r = get_redis()
        return r.get(step_name)

    @staticmethod
    def set_element(step_name, value):
        r = get_redis()
        return r.set(step_name, value)

    @staticmethod
    def delete(practice_name):
        r = get_redis()
        r.delete(practice_name)
