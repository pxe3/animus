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

        # 1 , generate initial reasoning given personality, context, and situation.
        reasoning = self._generate_initial_reasoning(
            situation=situation,
            traits=traits,
            context=context
        )

        # 2 , generate potential action based on reasoning
        action = self._generate_action(
            reasoning=reasoning,
            location=location
        )

        # 3 , think ahead
        next_steps = self._think_ahead(
            action=action,
            situation=situation,
            context=context
        )

        # 4 , assess confidence and social impact 
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
        And personality traits: {traits}
        With relevant past experiences: {context}
        
        Think through how to approach this situation, considering your personality
        and past experiences. What are the key factors to consider?
        """

        return self.llm.generate(
            prompt=prompt,
            temperature=0.7,
            verbose=True  # We'll see the full prompt/response
        )
    
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
        
        return self.llm.generate(
            prompt=prompt,
            temperature=0.7
        )

    def _think_ahead(self, action, situation, context, steps=3):
        """Think ahead about potential consequences and next steps (ToT-style)"""
        prompt = f"""Given the current situation:
        {situation}

        And the action you plan to take:
        {action}

        Consider the next {steps} likely outcomes or steps that could follow. For each step:
        1. What might happen next?
        2. How might others react?
        3. What follow-up actions might be needed?

        List exactly {steps} potential next steps in order, being specific and realistic.
        Format as a list with one step per line.
        """
        
        response = self.llm.generate(
            prompt=prompt,
            temperature=0.7
        )
        
        # Split into list and ensure we have exactly 'steps' number of items
        next_steps = [step.strip() for step in response.split('\n') if step.strip()]
        return next_steps[:steps]

    def _assess_confidence(self, reasoning, action, next_steps, traits):
        """Assess confidence in this thought path"""
        prompt = f"""Given this planned approach:
        
        Initial reasoning: {reasoning}
        Planned action: {action}
        Expected next steps: {', '.join(next_steps)}
        Your personality traits: {traits}

        Assess how confident you are in this approach succeeding.
        Consider:
        1. How well it aligns with your personality
        2. How realistic the expected outcomes are
        3. Potential complications or challenges
        4. Your ability to handle the next steps

        Return ONLY a confidence score between 0.0 and 1.0, where:
        0.0 = Complete uncertainty
        1.0 = Complete confidence
        
        Just the number, nothing else.
        """
        
        response = self.llm.generate(
            prompt=prompt,
            temperature=0.3  # Lower temperature for more consistent scoring
        )
        
        try:
            confidence = float(response.strip())
            return min(max(confidence, 0.0), 1.0)  # Clamp between 0 and 1
        except ValueError:
            return 0.5  # Default to medium confidence if parsing fails

    def _assess_social_impact(self, action, next_steps, context):    
        # First, get relationship effects in a controlled way
        relationships_prompt = f"""Analyze how this action will affect relationships:
        
        Action: {action}
        Expected steps: {', '.join(next_steps)}
        Context: {context}

        For each person mentioned, rate impact from -1 to 1.
        Respond ONLY with lines in this EXACT format:
        PERSON: SCORE
        Example:
        John: 0.5
        Mary: -0.3
        """
        
        relationship_response = self.llm.generate(
            prompt=relationships_prompt,
            temperature=0.3  # Lower temperature for more consistent output
        )
        
        relationship_effects = {}
        for line in relationship_response.split('\n'):
            if ':' in line:
                person, score = line.split(':')
                try:
                    relationship_effects[person.strip()] = min(max(float(score), -1.0), 1.0)
                except ValueError:
                    continue

        # Then get social standing separately
        standing_prompt = f"""Rate the overall social standing impact of this action:
        
        Action: {action}
        Context: {context}
        
        Respond with ONLY a number between -1 and 1:
        """
        
        try:
            social_standing = float(self.llm.generate(
                prompt=standing_prompt,
                temperature=0.3
            ).strip())
            social_standing = min(max(social_standing, -1.0), 1.0)
        except ValueError:
            social_standing = 0.0

        # Finally get risks in a structured way
        risks_prompt = f"""List the potential risks of this action.
        
        Action: {action}
        Context: {context}
        
        List EXACTLY 3 risks, one per line.
        Start each line with 'RISK: '
        """
        
        risks_response = self.llm.generate(
            prompt=risks_prompt,
            temperature=0.7
        )
        
        potential_risks = [
            line.replace('RISK: ', '').strip()
            for line in risks_response.split('\n')
            if line.startswith('RISK: ')
        ][:3]  # Take only first 3 risks

        return {
            "relationship_effects": relationship_effects,
            "social_standing": social_standing,
            "potential_risks": potential_risks
        }

    def execute(self, thought_path):    
        # First determine success/failure
        success_prompt = f"""Given this action:
        {thought_path.action}
        
        Was it successful? Respond with ONLY 'SUCCESS' or 'FAILURE':"""
        
        success = self.llm.generate(
            prompt=success_prompt,
            temperature=0.3
        ).strip().upper() == 'SUCCESS'

        # Then get specific outcome
        outcome_prompt = f"""Describe what specifically happened when this action was taken:
        {thought_path.action}
        
        Respond with a single clear sentence starting with 'OUTCOME: '"""
        
        outcome = self.llm.generate(
            prompt=outcome_prompt,
            temperature=0.7
        ).strip().replace('OUTCOME: ', '')

        # Get unexpected effects
        effects_prompt = f"""List any unexpected effects of this action:
        {thought_path.action}
        
        List up to 2 effects, one per line.
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

        # Reuse _assess_social_impact for actual impact calculation
        impact = self._assess_social_impact(
            thought_path.action,
            [],  # Empty next_steps since this is post-execution
            f"Action has already been executed. Outcome: {outcome}"
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
        success = result["success"]
        outcome = result["outcome"]
        actual_impact = result["actual_impact"]

        return self._calculate_score(success, outcome, actual_impact)

    def evaluate_thought_path(self, thought_path, situation, agent_traits):
        return {
            "score": self._evaluate_path_score(thought_path, situation, agent_traits),
            "reasoning": self._generate_evaluation_reasoning(thought_path, situation),
            "risks": self._identify_risks(thought_path),
            "opportunities": self._identify_opportunities(thought_path)
        }
    
    def _calculate_score(self, success, outcome, impact):
        base_score = 1.0 if success else 0.0
        
        # Adjust based on impact
        social_standing_modifier = impact.get("social_standing", 0) / 10.0
        relationship_modifier = self._calculate_relationship_modifier(
            impact.get("relationship_effects", {})
        )
        
        final_score = base_score + social_standing_modifier + relationship_modifier
        return min(max(final_score, 0.0), 1.0)  # Clamp between 0 and 1

    def _evaluate_path_score(self, thought_path, situation, agent_traits):
        prompt = f"""
        Given the situation: {situation}
        And personality traits: {agent_traits}
        
        Evaluate this potential approach:
        Reasoning: {thought_path.reasoning}
        Action: {thought_path.action}
        Expected next steps: {thought_path.next_steps}
        Social impact: {thought_path.social_impact}
        
        Rate this approach on:
        1. Alignment with personality
        2. Social appropriateness
        3. Likelihood of success
        4. Potential for positive outcomes
        """
        # TODO: Implement LLM evaluation
        return 0.7  # Placeholder score

    def _generate_evaluation_reasoning(self, thought_path, situation):
        # TODO: Implement LLM-based reasoning generation
        return "Evaluation reasoning placeholder"

    def _identify_risks(self, thought_path):
        # TODO: Implement risk identification
        return ["Risk 1", "Risk 2"]

    def _identify_opportunities(self, thought_path):
        # TODO: Implement opportunity identification
        return ["Opportunity 1", "Opportunity 2"]

    def _calculate_relationship_modifier(self, relationship_effects):
        if not relationship_effects:
            return 0.0
            
        # Average all relationship changes
        total_effect = sum(relationship_effects.values())
        return total_effect / len(relationship_effects) / 10.0  # Scale to [-0.1, 0.1]


class ReflectionModel:
    def __init__(self):
        self.llm = None

    def generate(self, situation, action, result, score, traits, location):
        """Generate a reflection on an experience that can be used for learning
        
        Args:
            situation (str): The original situation
            action (str): What the agent did
            result (dict): Outcome of the action
            score (float): Evaluation score from EvaluatorModel
            traits (dict): Agent's personality traits
            location (str): Where the interaction occurred
        
        Returns:
            dict: Reflection containing:
                - reflection (str): Verbal reflection on what happened
                - lessons (List[str]): Key lessons learned
                - future_strategies (List[str]): Strategies for similar situations
                - emotional_impact (dict): How this affected the agent
        """
        # Generate base reflection
        reflection = self._generate_base_reflection(
            situation=situation,
            action=action,
            result=result,
            score=score
        )

        # Extract key lessons
        lessons = self._extract_lessons(
            reflection=reflection,
            traits=traits
        )

        # Generate future strategies
        strategies = self._generate_future_strategies(
            reflection=reflection,
            lessons=lessons,
            traits=traits
        )

        # Assess emotional impact
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
        prompt = f"""
        Reflect on this social interaction:
        
        Situation: {situation}
        Action taken: {action}
        Outcome: {result['outcome']}
        Success level: {score}
        
        Provide a thoughtful reflection on:
        1. What went well
        2. What could have gone better
        3. Why things happened the way they did
        4. How this connects to past experiences
        """
        # TODO: Implement LLM reflection generation
        return "Base reflection placeholder"

    def _extract_lessons(self, reflection, traits):
        prompt = f"""
        Given this reflection:
        {reflection}
        
        And these personality traits:
        {traits}
        
        What are the key lessons to learn from this experience?
        Focus on lessons that align with the agent's personality.
        """
        # TODO: Implement LLM lesson extraction
        return [
            "Lesson 1",
            "Lesson 2",
            "Lesson 3"
        ]

    def _generate_future_strategies(self, reflection, lessons, traits):
        prompt = f"""
        Based on this reflection:
        {reflection}
        
        And these lessons learned:
        {lessons}
        
        Given personality traits:
        {traits}
        
        What strategies should be used in similar future situations?
        Consider both what worked well and what could be improved.
        """
        # TODO: Implement LLM strategy generation
        return [
            "Strategy 1",
            "Strategy 2",
            "Strategy 3"
        ]

    def _assess_emotional_impact(self, reflection, traits, result):
        prompt = f"""
        Given this reflection:
        {reflection}
        
        And these personality traits:
        {traits}
        
        How did this experience affect the agent emotionally?
        Consider both immediate and longer-term impact.
        """
        # TODO: Implement LLM emotional assessment
        return {
            "immediate_feelings": ["proud", "satisfied"],
            "confidence_change": 0.1,
            "relationship_impact": "strengthened",
            "long_term_effects": ["increased social confidence", "better understanding"]
        }