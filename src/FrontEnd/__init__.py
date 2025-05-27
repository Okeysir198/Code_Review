"""
Optimized FrontEnd package - Modular UI components for debt management.
"""

# Core interface components
from .front_end01_debtor_overview import create_client_overview_interface,navigate_to_client_details
from .front_end02_client_detail import create_client_details_interface
from .front_end20_audio_processing_tab import create_audio_processing_tab



__all__ = [
    # Main interfaces
    'create_client_overview_interface',
    'create_client_details_interface',
    'navigate_to_client_details',




    'create_audio_processing_tab'
    
]
