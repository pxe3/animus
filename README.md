
# 🤖 Social Intelligence in Agents

A framework combining Tree of Thoughts (ToT) and Reflexion architectures to create autonomous AI agents capable of thoughtful social interaction and learning from experiences.

## 🌟 Overview

This project creates socially intelligent AI agents by combining:
- **Tree of Thoughts**: Enables deliberate, multi-step reasoning for social decisions
- **Reflexion**: Provides memory and learning through verbal self-reflection
- **Social Context**: Considers personality traits, location, and relationship dynamics

Agents can:
- Generate and evaluate multiple possible paths of social interaction
- Think ahead about potential consequences
- Learn from past experiences through reflection
- Maintain personality consistency
- Develop relationships with other agents

## 💡 Features

- **Deliberate Decision Making**: Uses ToT to generate and evaluate multiple possible interaction paths
- **Experiential Learning**: Learns from interactions through Reflexion's verbal self-reflection
- **Memory Systems**: Maintains both short-term and long-term memory of experiences
- **Social Awareness**: Considers personality traits, location context, and relationship dynamics
- **Adaptive Behavior**: Improves social interactions over time through learning

# Generate social interaction
situation = "Meeting a new person at the cafe"
result = agent.think_and_act(situation, k=5)  # Generate 5 possible approaches
```

## 🔗 References

- [Tree of Thoughts: Deliberate Problem Solving with Large Language Models](https://arxiv.org/abs/2305.10601)
- [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)

