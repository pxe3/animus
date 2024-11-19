# agent.py
from .models import ActorModel, EvaluatorModel, ReflectionModel, ThoughtPath
from .memory import MemorySystem
from .llm import llm

class SocialAgent:    
    def __init__(self, name, traits, location=None):         
        self.name = name
        self.traits = traits
        self.location = location
        
        # Initialize all models
        self.actor = ActorModel()
        self.evaluator = EvaluatorModel()
        self.reflection = ReflectionModel()
        
        # Initialize memory systems
        self.memory = MemorySystem()
        self.short_term = []
        self.long_term = []
        
        # Current state
        self.current_activity = None
        self.current_interaction = None
        
    def think_and_act(self, situation, k=3):
        """Main decision loop combining ToT and Reflexion approaches"""
        # Get memory context
        memory_context = self._get_combined_context(situation)
        
        # Generate k thought paths using ActorModel's ToT
        thoughts = []
        for _ in range(k):
            thought = self.actor.generate(
                situation=situation,
                context=memory_context,
                traits=self.traits,
                location=self.location
            )
            thoughts.append(thought)
        
        # Evaluate and select best path
        best_path = self._select_best_path(thoughts, situation)
        
        # Execute chosen path
        result = self.actor.execute(best_path)
        
        # Update state
        self.current_activity = best_path.action
        self.current_interaction = self._extract_interaction(best_path.action)
        
        # Update memories and reflect
        self._update_memories(situation, best_path, result)
        
        return result

    def _get_combined_context(self, situation):
        """Combine recent context with relevant memories"""
        relevant_memories = self.memory.get_relevant_memories(
            situation=situation,
            k=3,
            current_location=self.location
        )
        
        # Get recent context
        recent_context = self.short_term[-3:] if self.short_term else []
        
        return self.memory.summarize_memories(relevant_memories) + "\n\nRecent events: " + str(recent_context)

    def _select_best_path(self, thoughts, situation):
        """Select best path using EvaluatorModel"""
        evaluations = [
            self.evaluator.evaluate_thought_path(
                thought_path=thought,
                situation=situation,
                agent_traits=self.traits
            )
            for thought in thoughts
        ]
        
        best_idx = max(range(len(evaluations)), 
                      key=lambda i: evaluations[i]["score"])
        return thoughts[best_idx]

    def _update_memories(self, situation, thought_path, result):
        """Update both memory systems after an action"""
        # Update short-term memory
        self.short_term.append({
            'situation': situation,
            'action': thought_path.action,
            'result': result,
            'location': self.location,
            'activity': self.current_activity,
            'interaction': self.current_interaction
        })
        
        # Keep short-term memory manageable
        if len(self.short_term) > 10:
            self.short_term = self.short_term[-10:]
        
        # Generate reflection
        reflection = self.reflection.generate(
            situation=situation,
            action=thought_path,
            result=result,
            score=self.evaluator.evaluate(result),
            traits=self.traits,
            location=self.location
        )
        
        # Store in long-term memory
        self.long_term.append(reflection)
        
        # Add to structured memory
        self.memory.add_memory(
            content=f"Action: {thought_path.action}\nOutcome: {result['outcome']}\nReflection: {reflection['reflection']}",
            memory_type="experience",
            location=self.location,
            people_involved=self._extract_people_from_situation(situation),
            emotional_impact=reflection["emotional_impact"].get("confidence_change", 0.0)
        )

    def _extract_interaction(self, action):
        if any(word in action.lower() for word in ["talk", "speak", "ask", "tell", "discuss"]):
            return "conversation"
        return None

    def _extract_people_from_situation(self, situation):
        prompt = f"""Who are the people mentioned in this situation?
        Return ONLY names, one per line:
        
        {situation}"""
        
        response = self.llm.generate(
            prompt=prompt,
            temperature=0.3
        )
        return [name.strip() for name in response.split('\n') if name.strip()]

    def move_to(self, new_location):
        self.location = new_location