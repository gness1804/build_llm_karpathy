"""
Utilities for parsing and extracting sections from v3 model responses.
"""

import re
from typing import Dict, Optional


def parse_v3_response(response: str) -> Dict[str, Optional[str]]:
    """
    Parse a v3 response into its component sections.
    
    Expected format:
    SCORE
    <score>
    
    STRENGTHS
    <strengths>
    
    WEAKNESSES
    <weaknesses>
    
    REVISED_RESPONSE
    <revised response>
    
    Args:
        response: The full response text from v3 model
        
    Returns:
        Dictionary with keys: 'score', 'strengths', 'weaknesses', 'revised_response'
        Values are None if section not found
    """
    result = {
        'score': None,
        'strengths': None,
        'weaknesses': None,
        'revised_response': None
    }
    
    # Pattern to match sections (case-insensitive, flexible whitespace)
    patterns = {
        'score': r'SCORE\s*\n\s*(.+?)(?=\n\s*(?:STRENGTHS|WEAKNESSES|REVISED_RESPONSE)|$)',
        'strengths': r'STRENGTHS\s*\n\s*(.+?)(?=\n\s*(?:WEAKNESSES|REVISED_RESPONSE|$))',
        'weaknesses': r'WEAKNESSES\s*\n\s*(.+?)(?=\n\s*REVISED_RESPONSE|$)',
        'revised_response': r'REVISED_RESPONSE\s*\n\s*(.+?)$'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE | re.MULTILINE)
        if match:
            result[key] = match.group(1).strip()
    
    return result


def extract_revised_response(response: str) -> str:
    """
    Extract just the REVISED_RESPONSE section from a v3 response.
    
    Args:
        response: The full response text from v3 model
        
    Returns:
        The revised response text, or the entire response if REVISED_RESPONSE section not found
    """
    parsed = parse_v3_response(response)
    
    if parsed['revised_response']:
        return parsed['revised_response']
    
    # Fallback: try to find REVISED_RESPONSE without strict formatting
    match = re.search(r'REVISED_RESPONSE[:\s]+\n?(.+?)$', response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Last resort: return entire response
    return response.strip()
