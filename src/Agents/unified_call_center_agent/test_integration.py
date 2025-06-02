# Integration test file: src/Agents/unified_call_center_agent/test_integration.py
"""
Integration tests for the unified call center agent
"""

def test_workflow_creation():
    """Test workflow creation"""
    from langchain_ollama import ChatOllama
    from .agent.unified_workflow import create_unified_call_center_workflow
    
    client_data = {
        'user_id': '12345',
        'profile': {
            'client_info': {
                'client_full_name': 'Test Client'
            }
        },
        'account_aging': {'xbalance': '100.00'}
    }
    
    model = ChatOllama(model="qwen2.5:7b-instruct", temperature=0)
    config = {'configurable': {'use_memory': False}}
    
    workflow = create_unified_call_center_workflow(model, client_data, config)
    assert workflow is not None
    print("✓ Workflow creation test passed")

def test_intent_detection():
    """Test intent detection"""
    from .routing.intent_detector import test_intent_detector
    
    results = test_intent_detector()
    accuracy = sum(1 for r in results if r["correct"]) / len(results)
    
    assert accuracy > 0.8, f"Intent detection accuracy too low: {accuracy:.2%}"
    print(f"✓ Intent detection test passed - Accuracy: {accuracy:.2%}")

def test_voice_integration():
    """Test voice integration"""
    from .integration.voice_integration import test_voice_integration
    
    voice_handler = test_voice_integration()
    assert voice_handler is not None
    print("✓ Voice integration test passed")

def run_all_tests():
    """Run all integration tests"""
    print("Running unified call center agent integration tests...")
    
    test_workflow_creation()
    test_intent_detection()
    test_voice_integration()
    
    print("All tests passed! ✅")

if __name__ == "__main__":
    run_all_tests()