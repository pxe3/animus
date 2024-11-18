# animus/tests/test_actor.py

from ..core.models import ActorModel, ThoughtPath
from ..core.llm import llm

def test_basic_social_scenario():
    actor = ActorModel()
    
    # Test data
    situation = """
    You're at a coffee shop working on your laptop. 
    You notice the person next to you seems to be struggling with coding a similar problem 
    you solved last week.
    """
    
    context = """
    You've helped other programmers before at this coffee shop.
    Last week you solved a similar coding problem.
    You generally enjoy helping others but are mindful of not being intrusive.
    """
    
    traits = {
        "helpful": 0.8,
        "empathetic": 0.7,
        "social": 0.6,
        "respectful": 0.9
    }
    
    location = "coffee_shop"

    print("\n=== TEST SCENARIO ===")
    print(f"Location: {location}")
    print(f"Traits: {traits}")
    print(f"Situation: {situation}")
    print(f"Context: {context}")

    # Generate thought path
    thought_path = actor.generate(
        situation=situation,
        context=context,
        traits=traits,
        location=location
    )

    print("\n=== THOUGHT PROCESS ===")
    print(f"Initial Reasoning:\n{thought_path.reasoning}")
    print(f"\nChosen Action:\n{thought_path.action}")
    
    print("\nAnticipated Next Steps:")
    for i, step in enumerate(thought_path.next_steps, 1):
        print(f"{i}. {step}")
    
    print(f"\nConfidence Score: {thought_path.confidence}")
    
    print("\nSocial Impact Analysis:")
    print(f"Relationship Effects: {thought_path.social_impact['relationship_effects']}")
    print(f"Social Standing: {thought_path.social_impact['social_standing']}")
    print("\nPotential Risks:")
    for risk in thought_path.social_impact['potential_risks']:
        print(f"- {risk}")

    # Execute the thought path
    result = actor.execute(thought_path)
    
    print("\n=== EXECUTION RESULTS ===")
    print(f"Success: {result['success']}")
    print(f"Outcome: {result['outcome']}")
    print("\nUnexpected Effects:")
    for effect in result['actual_impact']['unexpected_effects']:
        print(f"- {effect}")

    print("\n=== DECISION TREE SUMMARY ===")
    print("Initial Situation:")
    print("└── Reasoning:", thought_path.reasoning.replace('\n', '\n    '))
    print("    └── Chosen Action:", thought_path.action.replace('\n', '\n        '))
    print("        └── Expected Next Steps:")
    for i, step in enumerate(thought_path.next_steps, 1):
        print(f"            {i}. {step}")
    print("\nActual Result:")
    print(f"└── Success: {result['success']}")
    print(f"    └── Outcome: {result['outcome']}")
    print("        └── Unexpected Effects:")
    for effect in result['actual_impact']['unexpected_effects']:
        print(f"            - {effect}")
def test_complex_social_scenario():
    actor = ActorModel()
    
    # Test data
    situation = """
    You're at a friend's dinner party. Your close friend Alex just made a controversial 
    political comment that's created visible tension. Some guests seem offended, 
    others are awkwardly quiet. The host, Sam, looks uncomfortable.
    """
    
    context = """
    You've known Alex for years and usually enjoy their company.
    You disagree with their political views but value the friendship.
    Sam worked hard to bring this diverse group together.
    Previous dinner parties have been enjoyable and drama-free.
    """
    
    traits = {
        "diplomatic": 0.9,
        "empathetic": 0.8,
        "conflict_averse": 0.6,
        "socially_aware": 0.85
    }
    
    location = "sam_house"

    print("\n=== TEST SCENARIO ===")
    print(f"Location: {location}")
    print(f"Traits: {traits}")
    print(f"Situation: {situation}")
    print(f"Context: {context}")

    # Generate thought path
    thought_path = actor.generate(
        situation=situation,
        context=context,
        traits=traits,
        location=location
    )

    print("\n=== THOUGHT PROCESS ===")
    print(f"Initial Reasoning:\n{thought_path.reasoning}")
    print(f"\nChosen Action:\n{thought_path.action}")
    
    print("\nAnticipated Next Steps:")
    for i, step in enumerate(thought_path.next_steps, 1):
        print(f"{i}. {step}")
    
    print(f"\nConfidence Score: {thought_path.confidence}")
    
    print("\nSocial Impact Analysis:")
    print(f"Relationship Effects: {thought_path.social_impact['relationship_effects']}")
    print(f"Social Standing: {thought_path.social_impact['social_standing']}")
    print("\nPotential Risks:")
    for risk in thought_path.social_impact['potential_risks']:
        print(f"- {risk}")

    # Execute the thought path
    result = actor.execute(thought_path)
    
    print("\n=== EXECUTION RESULTS ===")
    print(f"Success: {result['success']}")
    print(f"Outcome: {result['outcome']}")
    print("\nUnexpected Effects:")
    for effect in result['actual_impact']['unexpected_effects']:
        print(f"- {effect}")

    print("\n=== DECISION TREE SUMMARY ===")
    print("Initial Situation:")
    print("└── Reasoning:", thought_path.reasoning.replace('\n', '\n    '))
    print("    └── Chosen Action:", thought_path.action.replace('\n', '\n        '))
    print("        └── Expected Next Steps:")
    for i, step in enumerate(thought_path.next_steps, 1):
        print(f"            {i}. {step}")
    print("\nActual Result:")
    print(f"└── Success: {result['success']}")
    print(f"    └── Outcome: {result['outcome']}")
    print("        └── Unexpected Effects:")
    for effect in result['actual_impact']['unexpected_effects']:
        print(f"            - {effect}")

if __name__ == "__main__":
    print("Testing basic social scenario...")
    test_basic_social_scenario()
    
    print("\n" + "="*50 + "\n")
    
    print("Testing complex social scenario...")
    test_complex_social_scenario()