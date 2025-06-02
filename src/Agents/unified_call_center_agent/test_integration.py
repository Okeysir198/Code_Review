#!/usr/bin/env python3
"""
Complete Integration Test for Unified Call Center Agent
Run this to verify everything works correctly
"""

import sys
import os
import traceback
from typing import Dict, Any

def test_imports():
    """Test if all imports work correctly"""
    print("=== Testing Imports ===")
    
    # Test core imports
    try:
        from src.Agents.unified_call_center_agent.core.unified_agent_state import (
            UnifiedAgentState, ConversationObjective, ClientMood, VerificationStatus
        )
        print("✓ Core state imports successful")
    except Exception as e:
        print(f"✗ Core state imports failed: {e}")
        return False
    
    # Test agent imports
    try:
        from src.Agents.unified_call_center_agent.agent.unified_agent import UnifiedCallCenterAgent
        from src.Agents.unified_call_center_agent.agent.unified_workflow import create_unified_call_center_workflow
        print("✓ Agent imports successful")
    except Exception as e:
        print(f"✗ Agent imports failed: {e}")
        return False
    
    # Test routing imports
    try:
        from src.Agents.unified_call_center_agent.routing.intent_detector import FastIntentDetector
        print("✓ Routing imports successful")
    except Exception as e:
        print(f"✗ Routing imports failed: {e}")
        return False
    
    # Test conversation manager
    try:
        from src.Agents.unified_call_center_agent.core.conversation_manager import ConversationManager
        print("✓ Conversation manager imports successful")
    except Exception as e:
        print(f"✗ Conversation manager imports failed: {e}")
        return False
    
    # Test voice integration
    try:
        from src.Agents.unified_call_center_agent.integration.voice_integration import (
            UnifiedVoiceHandler, VoiceInteractionHandler, create_unified_voice_handler
        )
        print("✓ Voice integration imports successful")
    except Exception as e:
        print(f"✗ Voice integration imports failed: {e}")
        return False
    
    # Test main package imports
    try:
        from src.Agents.unified_call_center_agent import (
            create_call_center_agent, VoiceInteractionHandler
        )
        print("✓ Main package imports successful")
    except Exception as e:
        print(f"✗ Main package imports failed: {e}")
        return False
    
    return True

def test_intent_detection():
    """Test intent detection functionality"""
    print("\n=== Testing Intent Detection ===")
    
    try:
        from src.Agents.unified_call_center_agent.routing.intent_detector import (
            FastIntentDetector, test_intent_detector
        )
        
        # Run built-in tests
        results = test_intent_detector()
        accuracy = sum(1 for r in results if r["correct"]) / len(results)
        
        if accuracy > 0.7:
            print(f"✓ Intent detection test passed - Accuracy: {accuracy:.2%}")
            return True
        else:
            print(f"✗ Intent detection accuracy too low: {accuracy:.2%}")
            return False
            
    except Exception as e:
        print(f"✗ Intent detection test failed: {e}")
        return False

def test_state_management():
    """Test state management functionality"""
    print("\n=== Testing State Management ===")
    
    try:
        from src.Agents.unified_call_center_agent.core.unified_agent_state import (
            UnifiedAgentState, ConversationObjective, ClientMood
        )
        
        # Create test client data
        test_client_data = {
            'user_id': '12345',
            'profile': {
                'user_id': '12345',
                'client_info': {
                    'client_full_name': 'Test Client',
                    'first_name': 'Test',
                    'title': 'Mr'
                }
            },
            'account_aging': {
                'xbalance': '299.00',
                'x0': '0.00',
                'x30': '299.00'
            }
        }
        
        # Test state creation
        state = UnifiedAgentState.create_initial_state(test_client_data)
        
        assert state.user_id == '12345'
        assert state.client_name == 'Test Client'
        assert state.outstanding_amount == 'R 299.00'
        assert state.current_objective == ConversationObjective.IDENTITY_VERIFICATION.value
        
        # Test state methods
        assert not state.is_verified()
        assert not state.can_discuss_account()
        
        context = state.get_conversation_context()
        assert isinstance(context, dict)
        assert 'client_mood' in context
        
        print("✓ State management test passed")
        return True
        
    except Exception as e:
        print(f"✗ State management test failed: {e}")
        traceback.print_exc()
        return False

def test_conversation_manager():
    """Test conversation manager functionality"""
    print("\n=== Testing Conversation Manager ===")
    
    try:
        from src.Agents.unified_call_center_agent.core.conversation_manager import ConversationManager
        from src.Agents.unified_call_center_agent.core.unified_agent_state import UnifiedAgentState
        from src.Agents.unified_call_center_agent.routing.intent_detector import FastIntentDetector, IntentMatch
        
        # Create test data
        test_client_data = {
            'user_id': '12345',
            'profile': {
                'client_info': {
                    'client_full_name': 'Test Client'
                }
            },
            'account_aging': {'xbalance': '100.00'}
        }
        
        # Create conversation manager
        conv_manager = ConversationManager(test_client_data)
        
        # Create test state
        state = UnifiedAgentState.create_initial_state(test_client_data)
        
        # Test getting next action
        action = conv_manager.get_next_action(state)
        
        assert isinstance(action, dict)
        assert 'action' in action
        assert 'objective' in action or 'message' in action
        
        print("✓ Conversation manager test passed")
        return True
        
    except Exception as e:
        print(f"✗ Conversation manager test failed: {e}")
        traceback.print_exc()
        return False

def test_unified_agent():
    """Test unified agent functionality"""
    print("\n=== Testing Unified Agent ===")
    
    try:
        # Mock ChatOllama for testing
        class MockChatModel:
            def invoke(self, messages):
                class MockResponse:
                    content = "Hello, I understand you're calling about your account."
                return MockResponse()
        
        from src.Agents.unified_call_center_agent.agent.unified_agent import UnifiedCallCenterAgent
        from src.Agents.unified_call_center_agent.core.unified_agent_state import UnifiedAgentState
        
        # Create test data
        test_client_data = {
            'user_id': '12345',
            'profile': {
                'client_info': {
                    'client_full_name': 'Test Client'
                }
            },
            'account_aging': {'xbalance': '100.00'}
        }
        
        test_config = {'configurable': {'use_memory': False}}
        
        # Create unified agent
        agent = UnifiedCallCenterAgent(
            model=MockChatModel(),
            client_data=test_client_data,
            config=test_config,
            agent_name="TestAgent"
        )
        
        # Create test state
        state = UnifiedAgentState.create_initial_state(test_client_data)
        state.turn_count = 1
        
        # Test conversation turn processing
        result = agent.process_conversation_turn(state)
        
        assert isinstance(result, dict)
        assert 'turn_count' in result
        assert 'last_action' in result
        
        print("✓ Unified agent test passed")
        return True
        
    except Exception as e:
        print(f"✗ Unified agent test failed: {e}")
        traceback.print_exc()
        return False

def test_workflow_creation():
    """Test workflow creation"""
    print("\n=== Testing Workflow Creation ===")
    
    try:
        # Mock ChatOllama for testing
        class MockChatModel:
            def invoke(self, messages):
                class MockResponse:
                    content = "Test response"
                return MockResponse()
        
        from src.Agents.unified_call_center_agent.agent.unified_workflow import create_unified_call_center_workflow
        
        # Create test data
        test_client_data = {
            'user_id': '12345',
            'profile': {
                'client_info': {
                    'client_full_name': 'Test Client'
                }
            },
            'account_aging': {'xbalance': '100.00'}
        }
        
        test_config = {'configurable': {'use_memory': False}}
        
        # Create workflow
        workflow = create_unified_call_center_workflow(
            model=MockChatModel(),
            client_data=test_client_data,
            config=test_config
        )
        
        assert workflow is not None
        
        # Test workflow invocation
        result = workflow.invoke({})
        assert isinstance(result, dict)
        
        print("✓ Workflow creation test passed")
        return True
        
    except Exception as e:
        print(f"✗ Workflow creation test failed: {e}")
        traceback.print_exc()
        return False

def test_voice_integration():
    """Test voice integration"""
    print("\n=== Testing Voice Integration ===")
    
    try:
        from src.Agents.unified_call_center_agent.integration.voice_integration import (
            UnifiedVoiceHandler, create_unified_voice_handler
        )
        
        # Test configuration
        test_config = {
            'llm': {
                'model_name': 'qwen2.5:7b-instruct',
                'temperature': 0
            },
            'configurable': {
                'use_memory': False,
                'enable_stt_model': False,  # Disable for testing
                'enable_tts_model': False   # Disable for testing
            }
        }
        
        # Mock workflow factory
        def mock_workflow_factory(client_data):
            class MockWorkflow:
                def invoke(self, input, config=None):
                    from langchain_core.messages import AIMessage
                    return {
                        'messages': [AIMessage(content="Test response")],
                        'user_id': client_data.get('user_id', ''),
                        'client_name': 'Test Client'
                    }
                
                def get_state(self, config):
                    class MockState:
                        values = {
                            'current_objective': 'identity_verification',
                            'turn_count': 1
                        }
                    return MockState()
            
            return MockWorkflow()
        
        # Create voice handler
        voice_handler = UnifiedVoiceHandler(test_config, mock_workflow_factory)
        
        # Test client data update
        test_client_data = {
            'user_id': '12345',
            'profile': {
                'client_info': {
                    'client_full_name': 'Test Client'
                }
            }
        }
        
        voice_handler.update_client_data('12345', test_client_data)
        
        # Test message processing
        result = voice_handler.process_message("Hello")
        assert isinstance(result, dict)
        
        print("✓ Voice integration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Voice integration test failed: {e}")
        traceback.print_exc()
        return False

def test_frontend_compatibility():
    """Test frontend compatibility"""
    print("\n=== Testing Frontend Compatibility ===")
    
    try:
        # Test that the VoiceInteractionHandler wrapper works
        from src.Agents.unified_call_center_agent.integration.voice_integration import VoiceInteractionHandler
        
        test_config = {
            'configurable': {
                'enable_stt_model': False,
                'enable_tts_model': False
            }
        }
        
        # Mock workflow factory
        def mock_workflow_factory(client_data):
            class MockWorkflow:
                def invoke(self, input, config=None):
                    from langchain_core.messages import AIMessage
                    return {'messages': [AIMessage(content="Test response")]}
            return MockWorkflow()
        
        # Create handler (should use compatibility wrapper)
        handler = VoiceInteractionHandler(test_config, mock_workflow_factory)
        
        # Test that it has the expected interface
        assert hasattr(handler, 'update_client_data')
        assert hasattr(handler, 'process_message')
        assert hasattr(handler, 'add_message_handler')
        
        print("✓ Frontend compatibility test passed")
        return True
        
    except Exception as e:
        print(f"✗ Frontend compatibility test failed: {e}")
        traceback.print_exc()
        return False

def test_package_imports():
    """Test main package imports work"""
    print("\n=== Testing Package-Level Imports ===")
    
    try:
        # Test main package imports
        from src.Agents.unified_call_center_agent import (
            create_call_center_agent,
            VoiceInteractionHandler,
            UnifiedAgentState,
            ConversationObjective
        )
        
        print("✓ Package-level imports test passed")
        return True
        
    except Exception as e:
        print(f"✗ Package-level imports test failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all integration tests"""
    print("🚀 Running Unified Call Center Agent Integration Tests")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_intent_detection,
        test_state_management,
        test_conversation_manager,
        test_unified_agent,
        test_workflow_creation,
        test_voice_integration,
        test_frontend_compatibility,
        test_package_imports
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The unified agent is ready to use.")
        print("\n🔧 Integration Instructions:")
        print("1. Ensure all __init__.py files are created")
        print("2. Update voice_chat_test_app.py with the frontend fix")
        print("3. Use 'from src.Agents.unified_call_center_agent import VoiceInteractionHandler'")
        print("4. No other changes needed - same interface as before!")
        return True
    else:
        print("❌ Some tests failed. Check the errors above.")
        return False

def create_file_structure():
    """Create the necessary file structure"""
    print("\n=== Creating File Structure ===")
    
    directories = [
        "src/Agents/unified_call_center_agent",
        "src/Agents/unified_call_center_agent/agent",
        "src/Agents/unified_call_center_agent/core", 
        "src/Agents/unified_call_center_agent/routing",
        "src/Agents/unified_call_center_agent/integration"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        init_file = os.path.join(directory, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write(f'# {directory.replace("/", ".")} package\n')
            print(f"✓ Created {init_file}")
    
    print("✓ File structure created")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--create-structure":
        create_file_structure()
    else:
        success = run_all_tests()
        sys.exit(0 if success else 1)