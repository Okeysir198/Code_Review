# Example usage file: src/Agents/unified_call_center_agent/examples.py
"""
Usage examples for the unified call center agent
"""

def example_basic_usage():
    """Basic usage example"""
    
    from langchain_ollama import ChatOllama
    from . import create_call_center_agent
    
    # Sample client data
    client_data = {
        'user_id': '12345',
        'profile': {
            'user_id': '12345',
            'client_info': {
                'client_full_name': 'John Smith',
                'first_name': 'John',
                'title': 'Mr'
            }
        },
        'account_aging': {
            'xbalance': '299.00',
            'x0': '0.00', 
            'x30': '299.00'
        }
    }
    
    # Configuration
    config = {
        'configurable': {'use_memory': True}
    }
    
    # Create model and workflow
    model = ChatOllama(model="qwen2.5:7b-instruct", temperature=0)
    workflow = create_call_center_agent(
        model=model,
        client_data=client_data,
        config=config,
        agent_name="Sarah"
    )
    
    return workflow

def example_voice_integration():
    """Voice integration example"""
    
    from .integration import create_unified_voice_handler
    
    config = {
        "llm": {
            "model_name": "qwen2.5:14b-instruct",
            "temperature": 0,
            "context_window": 32000
        },
        "configurable": {
            "use_memory": True
        }
    }
    
    # Create voice handler
    voice_handler = create_unified_voice_handler(config)
    
    # Load client data
    client_data = {
        'user_id': '12345',
        'profile': {
            'client_info': {
                'client_full_name': 'Jane Doe',
                'first_name': 'Jane'
            }
        },
        'account_aging': {'xbalance': '199.00', 'x30': '199.00'}
    }
    
    voice_handler.update_client_data('12345', client_data)
    
    return voice_handler

def example_integration_with_existing_frontend():
    """Example of integrating with existing voice chat frontend"""
    
    # In your existing voice_chat_test_app.py, replace:
    # self.voice_handler = VoiceInteractionHandler(CONFIG, workflow_factory)
    
    # With:
    # from src.Agents.unified_call_center_agent import VoiceInteractionHandler
    # self.voice_handler = VoiceInteractionHandler(CONFIG, workflow_factory)
    
    # No other changes needed - same interface!
    
    pass

if __name__ == "__main__":
    # Run examples
    print("Creating basic workflow...")
    workflow = example_basic_usage()
    print("✓ Basic workflow created")
    
    print("Creating voice handler...")
    voice_handler = example_voice_integration()
    print("✓ Voice handler created")
    
    print("All examples completed successfully!")