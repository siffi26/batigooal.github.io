---
title: notebook for C++ Primer Plus - Part I
date: 2018-09-26 20:06:36
tags:
---

# 第三章 处理数据
本章将要学习的内容包括：
> C++变量命名的规则
> C++内置整型：unsigned long, long, unsigned int, int, unsigned short, short, char, unsigned char, signed char, bool
> C++11新增类型：unsigned long long, long long
> 表示各种整型的系统限制的climits文件
> 各种整型的数字字面值（常量）
> 使用const限定符来创建符号常量
> C++内置的浮点类型：float, double, long double
> 表示各种浮点类型的系统限制的cfloat文件
> 各种浮点类型的数字字面值
> C++的算术运算符
> 自动类型转换
> 强制类型转换

内置的C++类型可以分为两组：fundamental types（基本类型）和compound types（复合类型）。本章介绍的是基本类型中的整型和浮点型，以及它们的一些变形。然后介绍如何在C++中进行算术运算。最后，介绍了C++如何将一种类型转换成另一种类型。

## 3.1 简单变量

### 3.1.1 变量命名
