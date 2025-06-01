# evaluation/phoenix_setup.py
"""
Basic Phoenix evaluation setup for call center agent
Start simple, expand gradually
"""

import os
from typing import Dict, Any, List
import pandas as pd

# Phoenix imports (install with: pip install arize-phoenix)
try:
    import phoenix as px
    from phoenix.evals import (
        HalluccinationEvaluator,
        QAEvaluator,
        RelevanceEvaluator,
        run_evals
    )
    PHOENIX_AVAILABLE = True
except ImportError:
    print("⚠️ Phoenix not installed. Run: pip install arize-phoenix")
    PHOENIX_AVAILABLE = False

class CallCenterEvaluator:
    """Simple evaluator for call center conversations"""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
        self.conversations = []
        self.session = None
        
    def start_session(self, session_name: str = "call_center_eval"):
        """Start Phoenix session for tracking"""
        if not PHOENIX_AVAILABLE:
            print("Phoenix not available, using mock session")
            return
            
        self.session = px.launch_app()
        print(f"🚀 Phoenix session started: {session_name}")
        print(f"🌐 Phoenix UI: {self.session}")
    
    def add_conversation(self, conversation_data: Dict[str, Any]):
        """Add a conversation for evaluation"""
        self.conversations.append(conversation_data)
        print(f"📝 Added conversation: {len(self.conversations)} total")
    
    def prepare_evaluation_data(self) -> pd.DataFrame:
        """Convert conversations to DataFrame for Phoenix evaluation"""
        eval_data = []
        
        for idx, conv in enumerate(self.conversations):
            # Extract basic conversation info
            messages = conv.get('messages', [])
            client_data = conv.get('client_data', {})
            final_state = conv.get('final_state', {})
            
            # Create evaluation record
            record = {
                'conversation_id': f"conv_{idx}",
                'conversation_text': self._format_conversation(messages),
                'client_name': client_data.get('profile', {}).get('client_info', {}).get('client_full_name', 'Unknown'),
                'outstanding_amount': client_data.get('account_aging', {}).get('xbalance', '0.00'),
                'final_step': final_state.get('current_step', 'unknown'),
                'payment_secured': final_state.get('payment_secured', False),
                'call_ended': final_state.get('is_call_ended', False),
                'verification_completed': final_state.get('name_verification_status') == 'VERIFIED' and 
                                        final_state.get('details_verification_status') == 'VERIFIED'
            }
            eval_data.append(record)
        
        return pd.DataFrame(eval_data)
    
    def _format_conversation(self, messages: List) -> str:
        """Format conversation messages for evaluation"""
        formatted = []
        for msg in messages:
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                role = "Agent" if msg.type == "ai" else "Client"
                formatted.append(f"{role}: {msg.content}")
            elif isinstance(msg, dict):
                role = "Agent" if msg.get('type') == 'ai' else "Client"
                formatted.append(f"{role}: {msg.get('content', '')}")
        
        return "\n".join(formatted)

# Simple usage example
def create_basic_evaluator():
    """Create and return a basic evaluator instance"""
    evaluator = CallCenterEvaluator()
    return evaluator

if __name__ == "__main__":
    # Basic test
    evaluator = create_basic_evaluator()
    evaluator.start_session()
    print("✅ Basic Phoenix evaluator ready")