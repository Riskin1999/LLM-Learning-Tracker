---
name: 每日打卡
about: 大模型学习每日进度记录
title: "[打卡] 第X周 DayX - YYYY-MM-DD"
labels: 
  - 阶段1/阶段2/阶段3
  - 每日打卡
assignees: []
body:
  - type: input
    id: date
    attributes:
      label: 日期
      description: 填写学习当天的日期，格式为YYYY-MM-DD
      placeholder: 2026-01-07
    validations:
      required: true
  - type: input
    id: week
    attributes:
      label: 周次
      description: 填写当前学习周次，格式为「第X周 DayX」
      placeholder: 第1周 Day1
    validations:
      required: true
  - type: textarea
    id: tasks
    attributes:
      label: 今日完成任务（对应周计划）
      description: 逐条列出今日完成的核心学习任务，与周计划对应
      placeholder: |
        1. 线性代数：推导SVD分解步骤
        2. Python：手动实现Adam优化器
    validations:
      required: true
  - type: textarea
    id: problems
    attributes:
      label: 遇到的问题
      description: 如实记录今日学习中遇到的技术难点、报错等
      placeholder: SVD分解中特征值排序逻辑不清晰
    validations:
      required: false
  - type: textarea
    id: solutions
    attributes:
      label: 解决方法
      description: 记录问题的解决过程、参考资料或暂时未解决的原因
      placeholder: 查看李沐视频+推导3次公式，已理解
    validations:
      required: false
  - type: input
    id: duration
    attributes:
      label: 学习时长
      description: 今日实际投入的学习时长（单位：小时）
      placeholder: 2h
    validations:
      required: true
  - type: input
    id: code-link
    attributes:
      label: 今日代码链接
      description: 仓库内对应代码/笔记的相对路径
      placeholder: /code/week1/day1/adam.py
    validations:
      required: false
  - type: checkboxes
    id: progress
    attributes:
      label: 阶段进度
      description: 勾选今日完成的进度项
      options:
        - label: 已完成今日目标
        - label: 已整理笔记
        - label: 已提交代码
    validations:
      required: false
---
