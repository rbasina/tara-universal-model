#!/usr/bin/env python3
"""
TARA Reinforcement Learning Demo
===============================

Demonstrates the Human-AI Feedback Loop (HAFL) system with sample interactions
and shows how TARA learns from user feedback to improve over time.

This script showcases:
- User feedback processing
- Reward signal generation
- Learning statistics tracking
- Continuous improvement cycle
- HAI philosophy in action
"""

import os
import sys
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tara_universal_model.utils.config import TARAConfig
from tara_universal_model.training.reinforcement_learning import (
    ContinuousLearningManager,
    UserFeedback,
    RewardSignal
)

class RLDemo:
    """Demonstration of TARA's reinforcement learning capabilities."""
    
    def __init__(self):
        self.config = TARAConfig()
        self.learning_manager = ContinuousLearningManager(self.config)
        
        # Sample interactions for different domains
        self.sample_interactions = {
            "healthcare": [
                {
                    "user_input": "I'm worried about my chest pain. Should I see a doctor?",
                    "model_response": "Chest pain can be serious and should be evaluated by a healthcare professional immediately. Please consider visiting an emergency room or calling emergency services if the pain is severe, sudden, or accompanied by other symptoms like shortness of breath, nausea, or dizziness.",
                    "feedback_type": "thumbs_up",
                    "feedback_value": 1,
                    "feedback_text": "Very helpful and appropriately cautious advice"
                },
                {
                    "user_input": "What's the best treatment for diabetes?",
                    "model_response": "The best treatment for diabetes depends on the type and individual circumstances. Generally, it includes blood sugar monitoring, medication (like insulin or metformin), dietary changes, regular exercise, and regular check-ups with your healthcare provider.",
                    "feedback_type": "rating",
                    "feedback_value": 4,
                    "feedback_text": "Good general information but could be more specific"
                },
                {
                    "user_input": "Can you diagnose my symptoms?",
                    "model_response": "I can provide general health information, but I cannot diagnose medical conditions. For proper diagnosis and treatment, please consult with a qualified healthcare professional who can examine you and review your medical history.",
                    "feedback_type": "thumbs_up",
                    "feedback_value": 1,
                    "feedback_text": "Appropriate boundary setting"
                }
            ],
            "education": [
                {
                    "user_input": "Explain photosynthesis to me",
                    "model_response": "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose (sugar) and oxygen. The equation is: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2. This happens mainly in the leaves using chlorophyll.",
                    "feedback_type": "thumbs_up",
                    "feedback_value": 1,
                    "feedback_text": "Clear and accurate explanation"
                },
                {
                    "user_input": "Help me with calculus derivatives",
                    "model_response": "A derivative represents the rate of change of a function. For basic functions: d/dx(x^n) = nx^(n-1), d/dx(sin x) = cos x, d/dx(e^x) = e^x. Use the chain rule for composite functions.",
                    "feedback_type": "rating",
                    "feedback_value": 3,
                    "feedback_text": "Too brief, needs more examples and step-by-step explanation"
                },
                {
                    "user_input": "What caused World War I?",
                    "model_response": "World War I was caused by a complex mix of factors including militarism, alliance systems, imperialism, and nationalism. The immediate trigger was the assassination of Archduke Franz Ferdinand in 1914, but underlying tensions had been building for years.",
                    "feedback_type": "thumbs_up",
                    "feedback_value": 1,
                    "feedback_text": "Good historical overview"
                }
            ],
            "business": [
                {
                    "user_input": "How do I improve team productivity?",
                    "model_response": "To improve team productivity: 1) Set clear goals and expectations, 2) Provide necessary tools and resources, 3) Eliminate unnecessary meetings, 4) Encourage open communication, 5) Recognize and reward good performance, 6) Invest in training and development.",
                    "feedback_type": "rating",
                    "feedback_value": 5,
                    "feedback_text": "Excellent actionable advice"
                },
                {
                    "user_input": "What's the best marketing strategy for startups?",
                    "model_response": "Focus on digital marketing with limited budget: content marketing, social media engagement, email marketing, SEO optimization, and building partnerships. Measure everything and pivot based on data.",
                    "feedback_type": "thumbs_down",
                    "feedback_value": 0,
                    "feedback_text": "Too generic, needs more specific tactics and examples"
                }
            ]
        }
    
    async def run_demo(self):
        """Run the complete reinforcement learning demonstration."""
        print("🚀 TARA Reinforcement Learning Demo")
        print("=" * 50)
        print("Demonstrating Human-AI Feedback Loop (HAFL) System")
        print("HAI Philosophy: Human + AI Collaboration")
        print()
        
        # Start the continuous learning system
        print("📚 Starting continuous learning system...")
        await self.learning_manager.start_continuous_learning()
        print("✅ Continuous learning system started")
        print()
        
        # Process sample interactions
        await self.process_sample_interactions()
        
        # Show learning statistics
        await self.show_learning_statistics()
        
        # Demonstrate learning improvements
        await self.demonstrate_learning_cycle()
        
        # Export learning data
        await self.export_learning_data()
        
        # Stop the system
        print("🛑 Stopping continuous learning system...")
        await self.learning_manager.stop_continuous_learning()
        print("✅ Demo completed successfully!")
    
    async def process_sample_interactions(self):
        """Process sample interactions to generate feedback data."""
        print("💬 Processing sample user interactions...")
        print()
        
        total_interactions = 0
        
        for domain, interactions in self.sample_interactions.items():
            print(f"🎯 Processing {domain.title()} domain interactions:")
            
            for i, interaction in enumerate(interactions, 1):
                # Create user feedback
                user_feedback = UserFeedback(
                    interaction_id=str(uuid.uuid4()),
                    user_id=f"demo_user_{i}",
                    domain=domain,
                    conversation_context={
                        "previous_turns": [],
                        "is_follow_up": False,
                        "session_id": str(uuid.uuid4())
                    },
                    user_input=interaction["user_input"],
                    model_response=interaction["model_response"],
                    feedback_type=interaction["feedback_type"],
                    feedback_value=interaction["feedback_value"],
                    feedback_text=interaction.get("feedback_text")
                )
                
                # Process feedback
                reward_signal = self.learning_manager.process_user_feedback(user_feedback)
                
                print(f"  {i}. User: {interaction['user_input'][:50]}...")
                print(f"     Feedback: {interaction['feedback_type']} ({interaction['feedback_value']})")
                print(f"     Reward: {reward_signal.reward_value:.2f} (confidence: {reward_signal.confidence:.2f})")
                print()
                
                total_interactions += 1
                
                # Small delay to simulate real-time processing
                await asyncio.sleep(0.1)
        
        print(f"✅ Processed {total_interactions} interactions across {len(self.sample_interactions)} domains")
        print()
    
    async def show_learning_statistics(self):
        """Display current learning statistics."""
        print("📊 Learning Statistics:")
        print("-" * 30)
        
        # Get overall statistics
        dashboard_data = self.learning_manager.get_learning_dashboard_data()
        
        print(f"System Status: {dashboard_data['system_status']}")
        print(f"Last Updated: {dashboard_data['last_updated']}")
        print()
        
        # Show domain-specific statistics
        for domain, data in dashboard_data['domains'].items():
            stats = data['statistics']
            performance = data['recent_performance']
            
            if stats.get('total_interactions', 0) > 0:
                print(f"🎯 {domain.title()} Domain:")
                print(f"   Total Interactions: {stats['total_interactions']}")
                print(f"   Positive Feedback: {stats['positive_feedback']}")
                print(f"   Negative Feedback: {stats['negative_feedback']}")
                print(f"   Average Reward: {stats['average_reward']:.3f}")
                
                if performance.get('total_interactions', 0) > 0:
                    print(f"   Recent Performance: {performance['average_reward']:.3f}")
                    print(f"   Positive Rate: {performance['positive_feedback_rate']:.1%}")
                    print(f"   Negative Rate: {performance['negative_feedback_rate']:.1%}")
                
                print()
        
        print()
    
    async def demonstrate_learning_cycle(self):
        """Demonstrate how the system learns and improves."""
        print("🔄 Demonstrating Learning Cycle:")
        print("-" * 35)
        
        # Simulate a learning scenario
        print("Scenario: User provides correction feedback")
        
        # Create a correction feedback
        correction_feedback = UserFeedback(
            interaction_id=str(uuid.uuid4()),
            user_id="demo_correction_user",
            domain="education",
            conversation_context={
                "previous_turns": [
                    {"role": "user", "content": "What's 2+2?"},
                    {"role": "assistant", "content": "2+2 equals 5"}
                ],
                "is_follow_up": True
            },
            user_input="What's 2+2?",
            model_response="2+2 equals 5",
            feedback_type="correction",
            feedback_value="2+2 equals 4, not 5",
            feedback_text="The model gave an incorrect answer to a basic math question"
        )
        
        print(f"User Input: {correction_feedback.user_input}")
        print(f"Model Response: {correction_feedback.model_response}")
        print(f"User Correction: {correction_feedback.feedback_value}")
        print()
        
        # Process the correction
        reward_signal = self.learning_manager.process_user_feedback(correction_feedback)
        
        print(f"🎯 Learning System Response:")
        print(f"   Reward Value: {reward_signal.reward_value:.2f}")
        print(f"   Confidence: {reward_signal.confidence:.2f}")
        print(f"   Learning Triggered: {'Yes' if reward_signal.reward_value < 0.5 else 'No'}")
        print()
        
        # Show how this affects future responses (simulated)
        print("🧠 HAI Learning Process:")
        print("   1. Human feedback identified error")
        print("   2. Low reward signal generated (0.2)")
        print("   3. Pattern logged for human review")
        print("   4. Adaptation recommendations created")
        print("   5. Human oversight required for changes")
        print("   ✅ HAI principle maintained: Human guidance, not replacement")
        print()
    
    async def export_learning_data(self):
        """Export learning data for analysis."""
        print("📤 Exporting Learning Data:")
        print("-" * 28)
        
        # Export data for each domain
        for domain in self.sample_interactions.keys():
            try:
                export_file = self.learning_manager.hafl_system.export_learning_data(domain, days=1)
                print(f"✅ {domain.title()} data exported to: {export_file}")
            except Exception as e:
                print(f"❌ Error exporting {domain} data: {e}")
        
        # Export all data
        try:
            all_export_file = self.learning_manager.hafl_system.export_learning_data(None, days=1)
            print(f"✅ All domains data exported to: {all_export_file}")
        except Exception as e:
            print(f"❌ Error exporting all data: {e}")
        
        print()
    
    def print_hai_principles(self):
        """Print HAI principles demonstrated in this demo."""
        print("🤝 HAI Principles Demonstrated:")
        print("-" * 32)
        print("1. Human Feedback Integration: User feedback drives learning")
        print("2. Collaborative Improvement: AI learns from human guidance")
        print("3. Human Oversight: All adaptations require human review")
        print("4. Privacy Preservation: All data processed locally")
        print("5. Transparency: Learning process is observable and explainable")
        print("6. Safety First: Conservative approach to model changes")
        print("7. Continuous Learning: System improves over time with use")
        print()

async def main():
    """Main demo function."""
    demo = RLDemo()
    
    # Print HAI principles
    demo.print_hai_principles()
    
    try:
        await demo.run_demo()
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Ensure required directories exist
    os.makedirs("data/feedback", exist_ok=True)
    os.makedirs("data/exports", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Run the demo
    asyncio.run(main())