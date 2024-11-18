from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ThoughtPath:
    reasoning: str
    action: str
    next_steps: List[str]
    confidence: float
    social_impact: Optional[dict] = None



class ActorModel:
    def __init__(self):
        self.llm = None

    def generate(self, situation, context, traits, location):

        # 1 , generate initial reasoning given personality, context, and situation.
        reasoning = self._generate_initial_reasoning(
            situation=situation,
            traits=traits,
            context=context
        )

        # 2 , generate potential action based on reasoning

    def execute(self, thought_path):

        pass

class EvaluatorModel:
    def __init__(self):
        pass

    def evaluate(self, result):
        pass

class ReflectionModel:
    def __init__(self):
        pass

    def generate(self, situation, action, result, score, traits, location):
        pass