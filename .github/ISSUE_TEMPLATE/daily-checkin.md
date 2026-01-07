name: 每日打卡
about: 大模型学习每日进度记录
title: "[打卡] 第X周 DayX - YYYY-MM-DD"
labels: 阶段1/阶段2/阶段3, 每日打卡
assignees: ''

body:
  - type: input
    id: date
    attributes:
      label: 日期
      placeholder: 2026-01-07
    validations:
      required: true
  - type: input
    id: week
    attributes:
      label: 周次
      placeholder: 第1周 Day1
    validations:
      required: true
  - type: textarea
    id: tasks
    attributes:
      label: 今日完成任务（对应周计划）
      placeholder: |
        1. 线性代数：推导SVD分解步骤
        2. Python：手动实现Adam优化器
    validations:
      required: true
  - type: textarea
    id: problems
    attributes:
      label: 遇到的问题
      placeholder: SVD分解中特征值排序逻辑不清晰
  - type: textarea
    id: solutions
    attributes:
      label: 解决方法
      placeholder: 查看李沐视频+推导3次公式，已理解
  - type: input
    id: duration
    attributes:
      label: 学习时长
      placeholder: 2h
    validations:
      required: true
  - type: input
    id: code-link
    attributes:
      label: 今日代码链接
      placeholder: /code/week1/day1/adam.py
  - type: checkboxes
    id: progress
    attributes:
      label: 阶段进度
      options:
        - label: 已完成今日目标
        - label: 已整理笔记
        - label: 已提交代码
