from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from .llm import llm

@dataclass
class Memory:
    content: str
    timestamp: datetime
    memory_type: str              # 'observation', 'reflection', or 'interaction'
    location: str
    people_involved: List[str]    # Fixed field name
    emotional_impact: float       # -1.0 to 1.0
    importance: float            # 0.0 to 1.0
    last_accessed: datetime

class MemorySystem:
    def __init__(self):
        self.memories: List[Memory] = []
        self.llm = llm

    def add_memory(self, 
                  content: str,
                  memory_type: str,
                  location: str,
                  people_involved: List[str] = None,
                  emotional_impact: float = 0.0) -> None:
        importance = self._calculate_importance(content)
        
        memory = Memory(
            content=content,
            timestamp=datetime.now(),
            memory_type=memory_type,
            location=location,
            people_involved=people_involved or [],
            emotional_impact=emotional_impact,
            importance=importance,
            last_accessed=datetime.now()
        )
        
        self.memories.append(memory)

    def get_relevant_memories(self, 
                            situation: str,
                            k: int = 3,
                            current_location: Optional[str] = None) -> List[Memory]:
        scored_memories = []
        for memory in self.memories:
            score = self._calculate_relevance(
                situation=situation,
                memory=memory,
                current_location=current_location
            )
            scored_memories.append((score, memory))
            
        scored_memories.sort(reverse=True, key=lambda x: x[0])
        relevant_memories = [m for _, m in scored_memories[:k]]
        
        # Update last_accessed
        for memory in relevant_memories:
            memory.last_accessed = datetime.now()
            
        return relevant_memories

    def _calculate_importance(self, content: str) -> float:
        prompt = f"""Rate the importance of this memory (0.0 to 1.0):
        Memory: {content}
        Consider:
        1. Emotional significance
        2. Impact on relationships
        3. Learning value
        4. Future relevance
        Return ONLY a number:"""
        
        try:
            score = float(self.llm.generate(
                prompt=prompt,
                temperature=0.3
            ).strip())
            return min(max(score, 0.0), 1.0)
        except ValueError:
            return 0.5

    def _calculate_relevance(self, 
                           situation: str, 
                           memory: Memory,
                           current_location: Optional[str]) -> float:
        prompt = f"""Rate relevance of this memory to current situation (0.0 to 1.0):
        Current situation: {situation}
        Memory: {memory.content}
        Consider context similarity, people involved, location, and emotions.
        Return ONLY a number:"""
        
        try:
            base_score = float(self.llm.generate(
                prompt=prompt,
                temperature=0.3
            ).strip())
        except ValueError:
            base_score = 0.5

        # Apply modifiers
        modifiers = 1.0
        if current_location and current_location == memory.location:
            modifiers += 0.2
            
        hours_old = (datetime.now() - memory.timestamp).total_seconds() / 3600
        if hours_old <= 24:
            modifiers += 0.2 * (1 - hours_old/24)
            
        modifiers += memory.importance * 0.2
        
        return min(max(base_score * modifiers, 0.0), 1.0)

    def summarize_memories(self, memories: List[Memory]) -> str:
        if not memories:
            return "No relevant memories."
            
        memory_texts = [m.content for m in memories]
        
        prompt = f"""Summarize these related memories into a coherent narrative:
        {chr(10).join(f"- {m}" for m in memory_texts)}
        Provide a brief summary capturing key points and relationships."""
        
        return self.llm.generate(
            prompt=prompt,
            temperature=0.7
        )

    def get_memories_about_person(self, person: str, k: int = 3) -> List[Memory]:
        relevant_memories = [
            m for m in self.memories 
            if person in m.people_involved
        ]
        
        scored_memories = []
        for memory in relevant_memories:
            hours_old = (datetime.now() - memory.timestamp).total_seconds() / 3600
            recency_score = 1.0 / (1.0 + hours_old/24)
            score = memory.importance * recency_score
            scored_memories.append((score, memory))
            
        scored_memories.sort(reverse=True, key=lambda x: x[0])
        return [m for _, m in scored_memories[:k]]