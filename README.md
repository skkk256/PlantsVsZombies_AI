# PythonPlantsVsZombies with AI

Game part fork from [PythonPlantsVsZombies](https://github.com/marblexu/PythonPlantsVsZombies)

Reference: https://hanadyg.github.io/portfolio/report/INF581_report.pdf

Proposal report: https://docs.qq.com/slide/DZk9qTk9vWWJKS29Y

## TODO
- [ ] 分成 GameState, Agent, GameRunner 三部分
- [ ] 四种植物: 豌豆射手、向日葵、土豆地雷、坚果墙, 僵尸只有血量和行动速度区别, state ( 每个格子上的植物、每个格子上的僵尸血量总和、可用植物状态、阳光数 )
  - logic-based method
  - Q-learning
  - [Deep Q-network](https://hanadyg.github.io/portfolio/report/INF581_report.pdf)
- [ ] [option] 更多植物和更多种类僵尸
- [ ] [option] 如何更优地选择植物

## 游戏部分
```
├── state
│   ├── levle.py     # 游戏界面
│   ├── mainmenu.py  # 开始界面
│   ├── screen.py    # 结束界面
│    ...
├── tool.py          # controler
```

## AI Agent
```
├── agents
│   ├── env.py       # 环境状态，游戏更新相关
│   ├── agent.py     # 智能体实现
│   ├── 
│    ...

```

## 已知 Bug
- 土豆地雷不会爆炸 ( 原版就有，pygame碰撞检测问题 )


