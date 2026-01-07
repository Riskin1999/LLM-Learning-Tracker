name: 每周复盘
about: 大模型学习每周总结
title: "[复盘] 第X周 - 阶段X"
labels: 阶段1/阶段2/阶段3, 每周复盘
assignees: ''

body:
  - type: input
    id: week
    attributes:
      label: 周次
      placeholder: 第1周
    validations:
      required: true
  - type: textarea
    id: summary
    attributes:
      label: 本周完成情况
      placeholder: 完成线性代数+概率论+优化理论学习，实现Adam优化器
    validations:
      required: true
  - type: textarea
    id: unfinish
    attributes:
      label: 未完成任务
      placeholder: 暂无
  - type: textarea
    id: improvements
    attributes:
      label: 改进计划
      placeholder: 下周加强Transformer代码实操
  - type: input
    id: next-week
    attributes:
      label: 下周核心目标
      placeholder: 掌握Transformer架构并实现Encoder
    validations:
      required: true
