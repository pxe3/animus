# models.py

from dataclasses import dataclass
from typing import List, Optional
from .llm import llm

__all__ = ['ActorModel', 'EvaluatorModel', 'ReflectionModel', 'ThoughtPath']

@dataclass
class ThoughtPath:
    reasoning: str
    action: str
    next_steps: List[str]
    confidence: float
    social_impact: Optional[dict] = None

class ActorModel:
    def __init__(self):
        self.llm = llm

    def generate(self, situation, context, traits, location):
        reasoning = self._generate_initial_reasoning(
            situation=situation,
            traits=traits,
            context=context
        )

        action = self._generate_action(
            reasoning=reasoning,
            location=location
        )

        next_steps = self._think_ahead(
            action=action,
            situation=situation,
            context=context
        )

        confidence = self._assess_confidence(
            reasoning=reasoning,
            action=action,
            next_steps=next_steps,
            traits=traits
        )

        social_impact = self._assess_social_impact(
            action=action,
            next_steps=next_steps,
            context=context
        )
        
        return ThoughtPath(
            reasoning=reasoning,
            action=action,
            next_steps=next_steps,
            confidence=confidence,
            social_impact=social_impact
        )

    def _generate_initial_reasoning(self, situation, traits, context):
        prompt = f"""Given the situation: {situation}
        And personality traits ranging from 0.0 to 1.0: {traits}
        With relevant past experiences: {context}
        
        Think through how to approach this situation, considering your personality
        and past experiences. What are the key factors to consider?
        """
        return self.llm.generate(prompt=prompt, temperature=0.7)
    
    def _generate_action(self, reasoning, location):
        prompt = f"""Based on your reasoning:
        {reasoning}

        And your current location:
        {location}

        What specific action will you take? Consider:
        1. What would be most natural for your personality
        2. What's appropriate for this location
        3. How to best achieve your goals

        Describe your chosen action clearly and concisely.
        """
        return self.llm.generate(prompt=prompt, temperature=0.7)

    def _think_ahead(self, action, situation, context, steps=3):
        prompt = f"""Given the current situation:
        {situation}

        And the action you plan to take:
        {action}

        Consider the next {steps} likely outcomes or steps that could follow.
        List exactly {steps} potential next steps in order, being specific and realistic.
        Format as a list with one step per line.
        """
        response = self.llm.generate(prompt=prompt, temperature=0.7)
        next_steps = [step.strip() for step in response.split('\n') if step.strip()]
        return next_steps[:steps]

    def _assess_confidence(self, reasoning, action, next_steps, traits):
        prompt = f"""Given this planned approach:
        Initial reasoning: {reasoning}
        Planned action: {action}
        Expected next steps: {', '.join(next_steps)}
        Your personality traits: {traits}

        Assess how confident you are in this approach succeeding.
        Return ONLY a confidence score between 0.0 and 1.0.
        """
        response = self.llm.generate(prompt=prompt, temperature=0.3)
        try:
            confidence = float(response.strip())
            return min(max(confidence, 0.0), 1.0)
        except ValueError:
            return 0.5

    def _assess_social_impact(self, action, next_steps, context):
        # Get relationship effects
        relationships_prompt = f"""Analyze how this action will affect relationships:
        Action: {action}
        Expected steps: {', '.join(next_steps)}
        Context: {context}

        For each person mentioned, rate impact from -1 to 1.
        Format: PERSON: SCORE
        """
        relationship_response = self.llm.generate(
            prompt=relationships_prompt,
            temperature=0.3
        )
        
        relationship_effects = {}
        for line in relationship_response.split('\n'):
            try:
                if ':' in line:
                    # Strip any extra whitespace/characters
                    clean_line = line.strip()
                    person, score_str = clean_line.rsplit(':', 1)  # Split on last colon
                    person = person.strip()
                    try:
                        score = float(score_str)
                        relationship_effects[person] = min(max(score, -1.0), 1.0)
                    except ValueError:
                        continue
            except Exception:
                continue
                
        # If no valid relationships parsed, provide default
        if not relationship_effects:
            relationship_effects = {"Generic_Observer": 0.0}
        # Get social standing impact
        standing_prompt = f"""Rate the overall social standing impact of this action (-1 to 1):
        Action: {action}
        Context: {context}
        Return ONLY a number:
        """
        try:
            social_standing = float(self.llm.generate(
                prompt=standing_prompt,
                temperature=0.3
            ).strip())
            social_standing = min(max(social_standing, -1.0), 1.0)
        except ValueError:
            social_standing = 0.0

        # Get risks
        risks_prompt = f"""List EXACTLY 3 potential risks of this action:
        Action: {action}
        Context: {context}
        Format each line with 'RISK: '
        """
        risks_response = self.llm.generate(
            prompt=risks_prompt,
            temperature=0.7
        )
        potential_risks = [
            line.replace('RISK: ', '').strip()
            for line in risks_response.split('\n')
            if line.startswith('RISK: ')
        ][:3]

        return {
            "relationship_effects": relationship_effects,
            "social_standing": social_standing,
            "potential_risks": potential_risks
        }

    def execute(self, thought_path):
        # Check success/failure
        success_prompt = f"""Given this action:
        {thought_path.action}
        Was it successful? Respond ONLY with 'SUCCESS' or 'FAILURE':"""
        
        success = self.llm.generate(
            prompt=success_prompt,
            temperature=0.3
        ).strip().upper() == 'SUCCESS'

        # Get outcome
        outcome_prompt = f""" What specifically happened when this action was taken:
        {thought_path.action} with the situation
        Start response with 'OUTCOME: '"""
        
        outcome = self.llm.generate(
            prompt=outcome_prompt,
            temperature=0.7
        ).strip().replace('OUTCOME: ', '')

        # Get unexpected effects
        effects_prompt = f"""List up to 2 unexpected effects of this action:
        {thought_path.action} and the reasoning: {thought_path.reasoning}
        Start each line with 'EFFECT: '"""
        
        effects_response = self.llm.generate(
            prompt=effects_prompt,
            temperature=0.7
        )
        unexpected_effects = [
            line.replace('EFFECT: ', '').strip()
            for line in effects_response.split('\n')
            if line.startswith('EFFECT: ')
        ][:2]

        impact = self._assess_social_impact(
            thought_path.action,
            [],
            f"Action has been executed. Outcome: {outcome}"
        )
        
        return {
            "success": success,
            "outcome": outcome,
            "actual_impact": {
                "relationship_effects": impact["relationship_effects"],
                "social_standing": impact["social_standing"],
                "unexpected_effects": unexpected_effects
            }
        }

class EvaluatorModel:
    def __init__(self):
        self.llm = llm

    def evaluate(self, result):
        return self._calculate_score(
            result["success"],
            result["outcome"],
            result["actual_impact"]
        )

    def evaluate_thought_path(self, thought_path, situation, agent_traits):
        score = self._evaluate_path_score(thought_path, situation, agent_traits)
        reasoning = self._generate_evaluation_reasoning(thought_path, situation)
        risks = self._identify_risks(thought_path)
        opportunities = self._identify_opportunities(thought_path)
        
        return {
            "score": score,
            "reasoning": reasoning,
            "risks": risks,
            "opportunities": opportunities
        }
    
    def _calculate_score(self, success, outcome, impact):
        base_score = 1.0 if success else 0.0
        social_standing_modifier = impact.get("social_standing", 0) / 10.0
        relationship_modifier = self._calculate_relationship_modifier(
            impact.get("relationship_effects", {})
        )
        
        final_score = base_score + social_standing_modifier + relationship_modifier
        return min(max(final_score, 0.0), 1.0)

    def _evaluate_path_score(self, thought_path, situation, agent_traits):
        prompt = f"""Rate this approach (0.0 to 1.0):
        Situation: {situation}
        Traits: {agent_traits}
        Reasoning: {thought_path.reasoning}
        Action: {thought_path.action}
        Expected steps: {thought_path.next_steps}
        Social impact: {thought_path.social_impact}
        Consider: personality alignment, appropriateness, success likelihood, outcomes
        Return ONLY a number:"""
        
        try:
            score = float(self.llm.generate(prompt=prompt, temperature=0.3).strip())
            return min(max(score, 0.0), 1.0)
        except ValueError:
            return 0.7

    def _generate_evaluation_reasoning(self, thought_path, situation):
        prompt = f"""Evaluate this approach:
        Situation: {situation}
        Path: {thought_path}
        Explain why this approach would or wouldn't work well."""
        
        return self.llm.generate(prompt=prompt, temperature=0.7)

    def _identify_risks(self, thought_path):
        prompt = f"""List 3 specific risks for this approach:
        {thought_path}
        Format with 'RISK: ' prefix"""
        
        response = self.llm.generate(prompt=prompt, temperature=0.7)
        risks = [
            line.replace('RISK: ', '').strip()
            for line in response.split('\n')
            if line.startswith('RISK: ')
        ]
        return risks[:3]

    def _identify_opportunities(self, thought_path):
        prompt = f"""List 3 potential opportunities in this approach:
        {thought_path}
        Format with 'OPPORTUNITY: ' prefix"""
        
        response = self.llm.generate(prompt=prompt, temperature=0.7)
        opportunities = [
            line.replace('OPPORTUNITY: ', '').strip()
            for line in response.split('\n')
            if line.startswith('OPPORTUNITY: ')
        ]
        return opportunities[:3]

    def _calculate_relationship_modifier(self, relationship_effects):
        if not relationship_effects:
            return 0.0
        total_effect = sum(relationship_effects.values())
        return total_effect / len(relationship_effects) / 10.0

class ReflectionModel:
    def __init__(self):
        self.llm = llm  

    def generate(self, situation, action, result, score, traits, location):
        reflection = self._generate_base_reflection(
            situation=situation,
            action=action,
            result=result,
            score=score
        )

        lessons = self._extract_lessons(
            reflection=reflection,
            traits=traits
        )

        strategies = self._generate_future_strategies(
            reflection=reflection,
            lessons=lessons,
            traits=traits
        )

        emotional_impact = self._assess_emotional_impact(
            reflection=reflection,
            traits=traits,
            result=result
        )

        return {
            "reflection": reflection,
            "lessons": lessons,
            "future_strategies": strategies,
            "emotional_impact": emotional_impact
        }

    def _generate_base_reflection(self, situation, action, result, score):
        prompt = f"""Reflect on this interaction:
        Situation: {situation}
        Action taken: {action}
        Outcome: {result['outcome']}
        Success level: {score}
        
        Consider:
        1. What went well
        2. What could improve
        3. Why things happened this way
        4. Connections to past experiences"""
        
        return self.llm.generate(prompt=prompt, temperature=0.7)

    def _extract_lessons(self, reflection, traits):
        prompt = f"""Given this reflection:
        {reflection}
        And these traits: {traits}
        
        Extract 3 key lessons learned.
        Format with 'LESSON: ' prefix"""
        
        response = self.llm.generate(prompt=prompt, temperature=0.7)
        lessons = [
            line.replace('LESSON: ', '').strip()
            for line in response.split('\n')
            if line.startswith('LESSON: ')
        ]
        return lessons[:3]

    def _generate_future_strategies(self, reflection, lessons, traits):
        prompt = f"""Based on:
        Reflection: {reflection}
        Lessons: {lessons}
        Traits: {traits}
        
        Suggest 3 strategies for similar situations.
        Format with 'STRATEGY: ' prefix"""
        
        response = self.llm.generate(prompt=prompt, temperature=0.7)
        strategies = [
            line.replace('STRATEGY: ', '').strip()
            for line in response.split('\n')
            if line.startswith('STRATEGY: ')
        ]
        return strategies[:3]

    def _assess_emotional_impact(self, reflection, traits, result):
        prompt = f"""Assess emotional impact:
        Reflection: {reflection}
        Traits: {traits}
        Result: {result}
        
        Consider immediate and long-term effects.
        Format:
        FEELINGS: (comma-separated)
        CONFIDENCE: (number between -1 and 1)
        RELATIONSHIP: (strengthened/weakened/unchanged)
        EFFECTS: (comma-separated)"""
        
        response = self.llm.generate(prompt=prompt, temperature=0.7)
        
        # Parse response
        feelings = []
        confidence_change = 0.1
        relationship_impact = "unchanged"
        long_term_effects = []
        
        for line in response.split('\n'):
            if line.startswith('FEELINGS:'):
                feelings = [f.strip() for f in line.replace('FEELINGS:', '').split(',')]
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence_change = float(line.replace('CONFIDENCE:', '').strip())
                except ValueError:
                    pass
            elif line.startswith('RELATIONSHIP:'):
                relationship_impact = line.replace('RELATIONSHIP:', '').strip()
            elif line.startswith('EFFECTS:'):
                long_term_effects = [e.strip() for e in line.replace('EFFECTS:', '').split(',')]

        return {
            "immediate_feelings": feelings,
            "confidence_change": confidence_change,
            "relationship_impact": relationship_impact,
            "long_term_effects": long_term_effects
        }