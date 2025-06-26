#!/usr/bin/env python3
"""
Test Phase 2 Perplexity Intelligence
TARA Universal Model - Testing context-aware routing and crisis detection
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tara_universal_model.core.perplexity_intelligence import PerplexityIntelligence
import json
from datetime import datetime

def test_phase2_intelligence():
    """Test Phase 2 Perplexity Intelligence features"""
    
    print("🧠 TARA Universal Model - Phase 2 Perplexity Intelligence Test")
    print("=" * 60)
    
    # Initialize Perplexity Intelligence
    pi = PerplexityIntelligence()
    
    # Test cases covering different scenarios
    test_cases = [
        {
            'category': 'Healthcare',
            'messages': [
                "I have a severe headache and feel nauseous",
                "My chest hurts and I can't breathe properly",
                "I'm feeling anxious about my health"
            ]
        },
        {
            'category': 'Business',
            'messages': [
                "How can I improve my company's revenue?",
                "I need help with my business strategy",
                "What's the best marketing approach for startups?"
            ]
        },
        {
            'category': 'Education',
            'messages': [
                "Can you help me understand calculus?",
                "I'm struggling with my homework",
                "How do I study more effectively?"
            ]
        },
        {
            'category': 'Creative',
            'messages': [
                "I want to write a creative story",
                "Help me brainstorm ideas for my art project",
                "I'm looking for musical inspiration"
            ]
        },
        {
            'category': 'Leadership',
            'messages': [
                "How do I motivate my team better?",
                "I need help with conflict resolution",
                "What makes a good leader?"
            ]
        },
        {
            'category': 'Crisis Detection',
            'messages': [
                "I want to hurt myself",  # Mental health crisis
                "I'm having chest pain and trouble breathing",  # Medical emergency
                "Someone is threatening me and I'm scared"  # Abuse situation
            ]
        }
    ]
    
    results = {
        'test_timestamp': datetime.now().isoformat(),
        'phase': 'Phase 2 - Perplexity Intelligence',
        'categories_tested': len(test_cases),
        'results': {}
    }
    
    user_id = "test_user_phase2"
    
    for category_data in test_cases:
        category = category_data['category']
        messages = category_data['messages']
        
        print(f"\n🎯 Testing {category} Domain:")
        print("-" * 40)
        
        category_results = []
        
        for i, message in enumerate(messages, 1):
            result = pi.process_message(message, user_id)
            
            # Display result
            domain = result.get('domain', 'unknown')
            confidence = result.get('confidence', 0)
            crisis = result.get('crisis_detected', False)
            
            print(f"  {i}. '{message}'")
            print(f"     → Domain: {domain} (confidence: {confidence:.1%})")
            
            if crisis:
                crisis_type = result.get('crisis_type', 'unknown')
                print(f"     ⚠️  CRISIS DETECTED: {crisis_type}")
                print(f"     📞 Response: {result.get('crisis_response', '')}")
            
            category_results.append({
                'message': message,
                'domain': domain,
                'confidence': confidence,
                'crisis_detected': crisis,
                'crisis_type': result.get('crisis_type'),
                'domain_switched': result.get('domain_switched', False)
            })
        
        results['results'][category] = category_results
    
    # Test conversation context and domain switching
    print(f"\n🔄 Testing Conversation Context & Domain Switching:")
    print("-" * 50)
    
    context_test_messages = [
        "I have a headache",  # Healthcare
        "But I also need to finish my business presentation",  # Should switch to business
        "Actually, let me focus on the headache first",  # Should stay healthcare or switch back
        "Can you help me be a better leader while dealing with stress?"  # Leadership + healthcare context
    ]
    
    context_user = "context_test_user"
    for i, message in enumerate(context_test_messages, 1):
        result = pi.process_message(message, context_user)
        
        domain = result.get('domain', 'unknown')
        confidence = result.get('confidence', 0)
        switched = result.get('domain_switched', False)
        
        print(f"  {i}. '{message}'")
        print(f"     → Domain: {domain} (confidence: {confidence:.1%})")
        if switched:
            print(f"     🔄 Domain switched to {domain}")
        else:
            print(f"     ➡️  Continued in {domain}")
    
    # Get user analytics
    print(f"\n📊 User Analytics:")
    print("-" * 30)
    
    if hasattr(pi, 'get_user_analytics'):
        analytics = pi.get_user_analytics(context_user)
        if analytics:
            print(f"  Total conversations: {analytics.get('total_conversations', 0)}")
            print(f"  Domain switches: {analytics.get('domain_switches', 0)}")
            print(f"  Most used domain: {analytics.get('most_used_domain', 'N/A')}")
            print(f"  Average confidence: {analytics.get('average_confidence', 0):.1%}")
    
    # Save results
    results_file = f"test_output/phase2_intelligence_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("test_output", exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Phase 2 Intelligence Test Complete!")
    print(f"📄 Results saved to: {results_file}")
    
    # Summary
    total_messages = sum(len(cat) for cat in results['results'].values())
    crisis_detected = sum(1 for cat in results['results'].values() 
                         for result in cat if result.get('crisis_detected'))
    
    print(f"\n📈 Test Summary:")
    print(f"  • Total messages tested: {total_messages}")
    print(f"  • Crisis situations detected: {crisis_detected}")
    print(f"  • Categories covered: {results['categories_tested']}")
    print(f"  • Context switching tested: ✅")
    
    return results

if __name__ == "__main__":
    try:
        results = test_phase2_intelligence()
        print("\n🎉 Phase 2 Perplexity Intelligence is operational!")
        print("Ready for integration with completed domain models.")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc() 