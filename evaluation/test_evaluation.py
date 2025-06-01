# test_evaluation.py
"""
Simple test to evaluate a single conversation
"""

import sys
import os

# Add your project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your existing components
from test_graph.client_data import client_data
from src.Agents.graph_test_call_center import (
    simulation_cooperative, 
    graph_call_center_agent1
)

# Import our new evaluator
from evaluation.phoenix_setup import CallCenterEvaluator

def run_single_conversation_test():
    """Test a single conversation and evaluate it"""
    
    print("🚀 Starting single conversation test...")
    
    # 1. Create evaluator
    evaluator = CallCenterEvaluator()
    evaluator.start_session("test_session")
    
    # 2. Run a simple conversation
    print("💬 Running conversation...")
    
    # Start conversation
    initial_state = {
        "messages": [],
        "current_step": "introduction"
    }
    
    try:
        # Let agent introduce
        agent_response = graph_call_center_agent1.invoke(initial_state)
        
        # Simulate client response
        client_message = {"role": "human", "content": "Yes, this is John Smith"}
        
        # Continue conversation for a few turns
        conversation_state = {
            "messages": agent_response.get("messages", []) + [client_message]
        }
        
        # Agent responds to confirmation
        final_response = graph_call_center_agent1.invoke(conversation_state)
        
        print("✅ Conversation completed")
        
        # 3. Prepare conversation data for evaluation
        conversation_data = {
            "messages": final_response.get("messages", []),
            "client_data": client_data,
            "final_state": final_response
        }
        
        # 4. Add to evaluator
        evaluator.add_conversation(conversation_data)
        
        # 5. Basic analysis
        df = evaluator.prepare_evaluation_data()
        print("\n📊 Basic Evaluation Results:")
        print(df.to_string())
        
        return evaluator, df
        
    except Exception as e:
        print(f"❌ Error during conversation: {e}")
        return None, None

if __name__ == "__main__":
    evaluator, results = run_single_conversation_test()
    
    if results is not None:
        print("\n🎉 Evaluation test completed successfully!")
        print("\nNext steps:")
        print("1. Install Phoenix: pip install arize-phoenix")
        print("2. Add more sophisticated evaluations")
        print("3. Create batch testing")
    else:
        print("❌ Test failed - check setup")