#!/usr/bin/python
# -*- coding: utf-8 -*-
#cangye@hotmail.com
"""
=====================
python 类的使用
=====================
简要介绍python类的使用方式
"""
print(__doc__)


class test1():
    def __init__(self, num1 = 1):
        self.my_num = num1
    def __call__(self, num2 = 2):
        self.my_num = self.my_num + num2
        print("函数调用:", self.my_num)
    def func(self, num3 = 2):
        self.my_num = self.my_num + num3
        print("类方法：", self.my_num)

my_cls = test1(num1=6)
my_cls(3)
my_cls.__call__(num2=3)
my_cls.func(3)