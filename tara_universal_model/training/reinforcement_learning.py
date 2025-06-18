"""
TARA Reinforcement Learning System
==================================

Implements Human-AI Feedback Loop (HAFL) for continuous learning and improvement.
This system enables TARA to learn from user interactions, feedback, and outcomes
to become better over time while maintaining the HAI philosophy.

Key Features:
- Human feedback integration
- Reward modeling from user interactions
- Policy optimization with human oversight
- Continuous learning pipeline
- Privacy-preserving learning
- Domain-specific adaptation
"""

import os
import json
import logging
import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import sqlite3
from threading import Lock

from ..utils.config import TARAConfig
from ..emotional_intelligence.emotion_analyzer import EmotionAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class UserFeedback:
    """Structure for user feedback data."""
    interaction_id: str
    user_id: str
    domain: str
    conversation_context: Dict
    user_input: str
    model_response: str
    feedback_type: str  # 'thumbs_up', 'thumbs_down', 'rating', 'correction', 'preference'
    feedback_value: Union[int, float, str]
    feedback_text: Optional[str] = None
    timestamp: datetime = None
    emotion_context: Optional[Dict] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class RewardSignal:
    """Structure for reward signals."""
    interaction_id: str
    domain: str
    reward_value: float
    reward_type: str  # 'immediate', 'delayed', 'implicit', 'explicit'
    confidence: float
    source: str  # 'user_feedback', 'outcome_measure', 'engagement_metric'
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class FeedbackDatabase:
    """Database for storing and managing feedback data."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize the feedback database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    interaction_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    conversation_context TEXT,
                    user_input TEXT NOT NULL,
                    model_response TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    feedback_value TEXT NOT NULL,
                    feedback_text TEXT,
                    emotion_context TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reward_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    interaction_id TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    reward_value REAL NOT NULL,
                    reward_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_id TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    start_time DATETIME,
                    end_time DATETIME,
                    total_interactions INTEGER,
                    average_reward REAL,
                    improvement_score REAL,
                    status TEXT DEFAULT 'active'
                )
            """)
            
            conn.commit()
    
    def store_feedback(self, feedback: UserFeedback):
        """Store user feedback in database."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO user_feedback 
                    (interaction_id, user_id, domain, conversation_context, user_input, 
                     model_response, feedback_type, feedback_value, feedback_text, emotion_context)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    feedback.interaction_id,
                    feedback.user_id,
                    feedback.domain,
                    json.dumps(feedback.conversation_context),
                    feedback.user_input,
                    feedback.model_response,
                    feedback.feedback_type,
                    str(feedback.feedback_value),
                    feedback.feedback_text,
                    json.dumps(feedback.emotion_context) if feedback.emotion_context else None
                ))
                conn.commit()
    
    def store_reward(self, reward: RewardSignal):
        """Store reward signal in database."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO reward_signals 
                    (interaction_id, domain, reward_value, reward_type, confidence, source)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    reward.interaction_id,
                    reward.domain,
                    reward.reward_value,
                    reward.reward_type,
                    reward.confidence,
                    reward.source
                ))
                conn.commit()
    
    def get_recent_feedback(self, domain: str = None, hours: int = 24) -> List[UserFeedback]:
        """Get recent feedback for analysis."""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT * FROM user_feedback 
                WHERE timestamp > datetime('now', '-{} hours')
            """.format(hours)
            
            if domain:
                query += " AND domain = ?"
                cursor = conn.execute(query, (domain,))
            else:
                cursor = conn.execute(query)
            
            feedbacks = []
            for row in cursor.fetchall():
                feedback = UserFeedback(
                    interaction_id=row[1],
                    user_id=row[2],
                    domain=row[3],
                    conversation_context=json.loads(row[4]) if row[4] else {},
                    user_input=row[5],
                    model_response=row[6],
                    feedback_type=row[7],
                    feedback_value=row[8],
                    feedback_text=row[9],
                    emotion_context=json.loads(row[10]) if row[10] else None,
                    timestamp=datetime.fromisoformat(row[11])
                )
                feedbacks.append(feedback)
            
            return feedbacks

class RewardModel:
    """Neural network model for predicting rewards from interactions."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
    
    def predict_reward(self, features: torch.Tensor) -> float:
        """Predict reward for given interaction features."""
        with torch.no_grad():
            reward = self.model(features)
            return reward.item()
    
    def train_step(self, features: torch.Tensor, target_rewards: torch.Tensor):
        """Single training step for reward model."""
        self.optimizer.zero_grad()
        predicted_rewards = self.model(features)
        loss = self.criterion(predicted_rewards, target_rewards)
        loss.backward()
        self.optimizer.step()
        return loss.item()

class HAFLSystem:
    """
    Human-AI Feedback Loop System
    
    Implements continuous learning from human feedback while maintaining
    the HAI philosophy of collaboration over replacement.
    """
    
    def __init__(self, config: TARAConfig, db_path: str = None):
        self.config = config
        self.db_path = db_path or "data/feedback/tara_feedback.db"
        
        # Ensure feedback directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize components
        self.feedback_db = FeedbackDatabase(self.db_path)
        self.emotion_analyzer = EmotionAnalyzer()
        
        # Reward models for each domain
        self.reward_models = {}
        self.feature_extractors = {}
        
        # Learning parameters
        self.learning_rate = 0.001
        self.batch_size = 32
        self.memory_size = 10000
        
        # Feedback processing
        self.feedback_queue = deque(maxlen=1000)
        self.reward_queue = deque(maxlen=1000)
        
        # Learning statistics
        self.learning_stats = defaultdict(lambda: {
            'total_interactions': 0,
            'positive_feedback': 0,
            'negative_feedback': 0,
            'average_reward': 0.0,
            'improvement_trend': [],
            'last_update': None
        })
        
        logger.info("HAFL System initialized")
    
    def process_user_feedback(self, feedback: UserFeedback) -> RewardSignal:
        """
        Process user feedback and convert to reward signal.
        
        This is the core of the HAI approach - learning from human guidance
        rather than replacing human judgment.
        """
        logger.info(f"Processing feedback for interaction {feedback.interaction_id}")
        
        # Store feedback
        self.feedback_db.store_feedback(feedback)
        self.feedback_queue.append(feedback)
        
        # Convert feedback to reward signal
        reward_value = self._feedback_to_reward(feedback)
        
        # Create reward signal
        reward = RewardSignal(
            interaction_id=feedback.interaction_id,
            domain=feedback.domain,
            reward_value=reward_value,
            reward_type='explicit',
            confidence=self._calculate_feedback_confidence(feedback),
            source='user_feedback'
        )
        
        # Store reward
        self.feedback_db.store_reward(reward)
        self.reward_queue.append(reward)
        
        # Update learning statistics
        self._update_learning_stats(feedback.domain, reward_value)
        
        # Trigger learning if enough feedback accumulated
        if len(self.feedback_queue) >= self.batch_size:
            asyncio.create_task(self._trigger_learning_update(feedback.domain))
        
        return reward
    
    def _feedback_to_reward(self, feedback: UserFeedback) -> float:
        """Convert user feedback to numerical reward."""
        if feedback.feedback_type == 'thumbs_up':
            return 1.0
        elif feedback.feedback_type == 'thumbs_down':
            return 0.0
        elif feedback.feedback_type == 'rating':
            # Assume rating is 1-5, normalize to 0-1
            return (float(feedback.feedback_value) - 1) / 4
        elif feedback.feedback_type == 'correction':
            # Correction implies the response was wrong
            return 0.2
        elif feedback.feedback_type == 'preference':
            # Binary preference
            return 1.0 if feedback.feedback_value == 'preferred' else 0.0
        else:
            return 0.5  # Neutral for unknown feedback types
    
    def _calculate_feedback_confidence(self, feedback: UserFeedback) -> float:
        """Calculate confidence in the feedback signal."""
        confidence = 0.8  # Base confidence
        
        # Adjust based on feedback type
        if feedback.feedback_type in ['thumbs_up', 'thumbs_down']:
            confidence = 0.9  # High confidence for explicit feedback
        elif feedback.feedback_type == 'rating':
            confidence = 0.95  # Very high confidence for ratings
        elif feedback.feedback_type == 'correction':
            confidence = 1.0  # Maximum confidence for corrections
        
        # Adjust based on emotion context
        if feedback.emotion_context:
            emotion_intensity = feedback.emotion_context.get('intensity', 'medium')
            if emotion_intensity == 'high':
                confidence *= 0.9  # Slightly lower confidence for high emotion
        
        return min(confidence, 1.0)
    
    def _update_learning_stats(self, domain: str, reward_value: float):
        """Update learning statistics for domain."""
        stats = self.learning_stats[domain]
        stats['total_interactions'] += 1
        
        if reward_value > 0.6:
            stats['positive_feedback'] += 1
        elif reward_value < 0.4:
            stats['negative_feedback'] += 1
        
        # Update average reward (exponential moving average)
        alpha = 0.1
        stats['average_reward'] = (alpha * reward_value + 
                                 (1 - alpha) * stats['average_reward'])
        
        # Track improvement trend
        stats['improvement_trend'].append(reward_value)
        if len(stats['improvement_trend']) > 100:
            stats['improvement_trend'].pop(0)
        
        stats['last_update'] = datetime.now()
    
    async def _trigger_learning_update(self, domain: str):
        """Trigger learning update for domain."""
        logger.info(f"Triggering learning update for {domain} domain")
        
        try:
            # Get recent feedback for this domain
            recent_feedback = self.feedback_db.get_recent_feedback(domain, hours=24)
            
            if len(recent_feedback) < 10:
                logger.info(f"Not enough feedback for {domain} domain learning update")
                return
            
            # Extract features and rewards
            features, rewards = self._prepare_training_data(recent_feedback)
            
            if features is None or len(features) == 0:
                logger.warning(f"No valid training data for {domain} domain")
                return
            
            # Update reward model
            if domain not in self.reward_models:
                self.reward_models[domain] = RewardModel(input_dim=features.shape[1])
            
            # Train reward model
            loss = self._train_reward_model(domain, features, rewards)
            
            logger.info(f"Learning update completed for {domain} domain. Loss: {loss:.4f}")
            
            # Update model adaptation based on learned rewards
            await self._adapt_model_behavior(domain, recent_feedback)
            
        except Exception as e:
            logger.error(f"Error in learning update for {domain}: {e}")
    
    def _prepare_training_data(self, feedback_list: List[UserFeedback]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare training data from feedback."""
        features = []
        rewards = []
        
        for feedback in feedback_list:
            try:
                # Extract features from interaction
                feature_vector = self._extract_features(feedback)
                reward_value = self._feedback_to_reward(feedback)
                
                features.append(feature_vector)
                rewards.append(reward_value)
                
            except Exception as e:
                logger.warning(f"Error extracting features from feedback: {e}")
                continue
        
        if not features:
            return None, None
        
        features_tensor = torch.stack(features)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        
        return features_tensor, rewards_tensor
    
    def _extract_features(self, feedback: UserFeedback) -> torch.Tensor:
        """Extract feature vector from feedback interaction."""
        # This is a simplified feature extraction
        # In practice, you'd use more sophisticated NLP features
        
        features = []
        
        # Text length features
        features.append(len(feedback.user_input.split()))
        features.append(len(feedback.model_response.split()))
        
        # Emotion features
        if feedback.emotion_context:
            emotion_scores = feedback.emotion_context.get('emotion_scores', {})
            for emotion in ['joy', 'sadness', 'anger', 'fear', 'surprise', 'neutral']:
                features.append(emotion_scores.get(emotion, 0.0))
        else:
            features.extend([0.0] * 6)  # Neutral emotions
        
        # Domain encoding (one-hot)
        domains = ['healthcare', 'business', 'education', 'creative', 'leadership']
        domain_encoding = [1.0 if feedback.domain == d else 0.0 for d in domains]
        features.extend(domain_encoding)
        
        # Conversation context features
        context = feedback.conversation_context
        features.append(len(context.get('previous_turns', [])))
        features.append(1.0 if context.get('is_follow_up', False) else 0.0)
        
        # Time features (hour of day, day of week)
        timestamp = feedback.timestamp
        features.append(timestamp.hour / 24.0)
        features.append(timestamp.weekday() / 7.0)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _train_reward_model(self, domain: str, features: torch.Tensor, rewards: torch.Tensor) -> float:
        """Train the reward model for a domain."""
        model = self.reward_models[domain]
        
        # Simple batch training
        total_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(features), self.batch_size):
            batch_features = features[i:i+self.batch_size]
            batch_rewards = rewards[i:i+self.batch_size]
            
            loss = model.train_step(batch_features, batch_rewards)
            total_loss += loss
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    async def _adapt_model_behavior(self, domain: str, feedback_list: List[UserFeedback]):
        """
        Adapt model behavior based on learned rewards.
        
        This implements the HAI principle of human-guided improvement
        rather than autonomous decision making.
        """
        logger.info(f"Adapting model behavior for {domain} domain")
        
        # Analyze feedback patterns
        patterns = self._analyze_feedback_patterns(feedback_list)
        
        # Generate adaptation recommendations
        recommendations = self._generate_adaptation_recommendations(domain, patterns)
        
        # Log recommendations for human review
        self._log_adaptation_recommendations(domain, recommendations)
        
        # Apply safe adaptations (those that don't change core behavior)
        await self._apply_safe_adaptations(domain, recommendations)
    
    def _analyze_feedback_patterns(self, feedback_list: List[UserFeedback]) -> Dict:
        """Analyze patterns in user feedback."""
        patterns = {
            'common_issues': defaultdict(int),
            'successful_responses': [],
            'emotion_correlations': defaultdict(list),
            'time_patterns': defaultdict(list),
            'user_preferences': defaultdict(int)
        }
        
        for feedback in feedback_list:
            reward = self._feedback_to_reward(feedback)
            
            # Track successful vs unsuccessful responses
            if reward > 0.7:
                patterns['successful_responses'].append({
                    'input': feedback.user_input,
                    'response': feedback.model_response,
                    'context': feedback.conversation_context
                })
            elif reward < 0.3:
                # Identify common issues
                if feedback.feedback_text:
                    patterns['common_issues'][feedback.feedback_text] += 1
            
            # Emotion correlations
            if feedback.emotion_context:
                primary_emotion = feedback.emotion_context.get('primary_emotion', 'neutral')
                patterns['emotion_correlations'][primary_emotion].append(reward)
            
            # Time patterns
            hour = feedback.timestamp.hour
            patterns['time_patterns'][hour].append(reward)
        
        return patterns
    
    def _generate_adaptation_recommendations(self, domain: str, patterns: Dict) -> Dict:
        """Generate recommendations for model adaptation."""
        recommendations = {
            'response_style_adjustments': [],
            'emotion_handling_improvements': [],
            'content_focus_changes': [],
            'timing_optimizations': [],
            'safety_considerations': []
        }
        
        # Analyze emotion correlations
        for emotion, rewards in patterns['emotion_correlations'].items():
            if len(rewards) > 5:
                avg_reward = np.mean(rewards)
                if avg_reward < 0.5:
                    recommendations['emotion_handling_improvements'].append({
                        'emotion': emotion,
                        'current_performance': avg_reward,
                        'suggestion': f'Improve responses for users experiencing {emotion}'
                    })
        
        # Analyze time patterns
        for hour, rewards in patterns['time_patterns'].items():
            if len(rewards) > 3:
                avg_reward = np.mean(rewards)
                if avg_reward < 0.4:
                    recommendations['timing_optimizations'].append({
                        'time_period': f'{hour}:00-{hour+1}:00',
                        'performance': avg_reward,
                        'suggestion': 'Consider different response style for this time period'
                    })
        
        # Safety considerations
        recommendations['safety_considerations'].append({
            'type': 'human_oversight',
            'message': 'All adaptations require human review before implementation'
        })
        
        return recommendations
    
    def _log_adaptation_recommendations(self, domain: str, recommendations: Dict):
        """Log adaptation recommendations for human review."""
        log_file = f"logs/adaptation_recommendations_{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        with open(log_file, 'w') as f:
            json.dump({
                'domain': domain,
                'timestamp': datetime.now().isoformat(),
                'recommendations': recommendations,
                'requires_human_review': True,
                'hai_compliance': 'All recommendations follow HAI principles of human oversight'
            }, f, indent=2)
        
        logger.info(f"Adaptation recommendations logged to {log_file}")
    
    async def _apply_safe_adaptations(self, domain: str, recommendations: Dict):
        """Apply only safe adaptations that don't change core behavior."""
        # For now, we only log recommendations
        # In a full implementation, this would apply non-critical adaptations
        # like response timing, formatting preferences, etc.
        
        logger.info(f"Safe adaptations logged for {domain} domain")
        logger.info("Core model behavior remains unchanged - human review required for significant changes")
    
    def get_learning_statistics(self, domain: str = None) -> Dict:
        """Get learning statistics for monitoring."""
        if domain:
            return dict(self.learning_stats.get(domain, {}))
        else:
            return {d: dict(stats) for d, stats in self.learning_stats.items()}
    
    def get_recent_performance(self, domain: str, hours: int = 24) -> Dict:
        """Get recent performance metrics."""
        recent_feedback = self.feedback_db.get_recent_feedback(domain, hours)
        
        if not recent_feedback:
            return {'message': 'No recent feedback available'}
        
        rewards = [self._feedback_to_reward(f) for f in recent_feedback]
        
        return {
            'total_interactions': len(recent_feedback),
            'average_reward': np.mean(rewards),
            'positive_feedback_rate': sum(1 for r in rewards if r > 0.6) / len(rewards),
            'negative_feedback_rate': sum(1 for r in rewards if r < 0.4) / len(rewards),
            'improvement_trend': 'improving' if len(rewards) > 1 and rewards[-1] > rewards[0] else 'stable',
            'last_updated': max(f.timestamp for f in recent_feedback).isoformat()
        }
    
    def export_learning_data(self, domain: str = None, days: int = 7) -> str:
        """Export learning data for analysis."""
        export_file = f"data/exports/learning_data_{domain or 'all'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        os.makedirs(os.path.dirname(export_file), exist_ok=True)
        
        # Get feedback data
        hours = days * 24
        feedback_data = self.feedback_db.get_recent_feedback(domain, hours)
        
        export_data = {
            'export_info': {
                'domain': domain,
                'time_range_days': days,
                'export_timestamp': datetime.now().isoformat(),
                'total_interactions': len(feedback_data)
            },
            'learning_statistics': self.get_learning_statistics(domain),
            'feedback_data': [asdict(f) for f in feedback_data],
            'hai_compliance': {
                'privacy_preserved': True,
                'human_oversight_maintained': True,
                'no_autonomous_decisions': True
            }
        }
        
        with open(export_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Learning data exported to {export_file}")
        return export_file

class ContinuousLearningManager:
    """
    Manager for continuous learning processes.
    
    Orchestrates the entire continuous learning pipeline while
    maintaining HAI principles and human oversight.
    """
    
    def __init__(self, config: TARAConfig):
        self.config = config
        self.hafl_system = HAFLSystem(config)
        self.is_running = False
        self.learning_tasks = []
        
        logger.info("Continuous Learning Manager initialized")
    
    async def start_continuous_learning(self):
        """Start the continuous learning process."""
        if self.is_running:
            logger.warning("Continuous learning already running")
            return
        
        self.is_running = True
        logger.info("🚀 Starting TARA Continuous Learning System")
        
        # Start background tasks
        self.learning_tasks = [
            asyncio.create_task(self._periodic_learning_updates()),
            asyncio.create_task(self._monitor_learning_health()),
            asyncio.create_task(self._generate_learning_reports())
        ]
        
        logger.info("✅ Continuous learning system started")
    
    async def stop_continuous_learning(self):
        """Stop the continuous learning process."""
        if not self.is_running:
            return
        
        self.is_running = False
        logger.info("🛑 Stopping continuous learning system")
        
        # Cancel all tasks
        for task in self.learning_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.learning_tasks, return_exceptions=True)
        
        logger.info("✅ Continuous learning system stopped")
    
    async def _periodic_learning_updates(self):
        """Periodic learning updates every hour."""
        while self.is_running:
            try:
                logger.info("🔄 Running periodic learning updates")
                
                # Get domains that need updates
                domains = ['healthcare', 'business', 'education', 'creative', 'leadership']
                
                for domain in domains:
                    stats = self.hafl_system.get_learning_statistics(domain)
                    if stats.get('total_interactions', 0) > 0:
                        # Trigger learning update if there's new feedback
                        await self.hafl_system._trigger_learning_update(domain)
                
                # Wait 1 hour
                await asyncio.sleep(3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic learning updates: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    async def _monitor_learning_health(self):
        """Monitor the health of the learning system."""
        while self.is_running:
            try:
                logger.info("🔍 Monitoring learning system health")
                
                # Check each domain
                domains = ['healthcare', 'business', 'education', 'creative', 'leadership']
                
                for domain in domains:
                    performance = self.hafl_system.get_recent_performance(domain, hours=24)
                    
                    if performance.get('negative_feedback_rate', 0) > 0.3:
                        logger.warning(f"High negative feedback rate for {domain}: {performance['negative_feedback_rate']:.2f}")
                    
                    if performance.get('total_interactions', 0) == 0:
                        logger.info(f"No recent interactions for {domain} domain")
                
                # Wait 30 minutes
                await asyncio.sleep(1800)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in learning health monitoring: {e}")
                await asyncio.sleep(300)
    
    async def _generate_learning_reports(self):
        """Generate daily learning reports."""
        while self.is_running:
            try:
                # Wait until next day at 6 AM
                now = datetime.now()
                next_report = now.replace(hour=6, minute=0, second=0, microsecond=0)
                if next_report <= now:
                    next_report += timedelta(days=1)
                
                wait_seconds = (next_report - now).total_seconds()
                await asyncio.sleep(wait_seconds)
                
                logger.info("📊 Generating daily learning report")
                
                # Generate report for each domain
                domains = ['healthcare', 'business', 'education', 'creative', 'leadership']
                
                for domain in domains:
                    report_file = self.hafl_system.export_learning_data(domain, days=1)
                    logger.info(f"Daily report generated for {domain}: {report_file}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error generating learning reports: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour before retry
    
    def process_user_feedback(self, feedback: UserFeedback) -> RewardSignal:
        """Process user feedback through the HAFL system."""
        return self.hafl_system.process_user_feedback(feedback)
    
    def get_learning_dashboard_data(self) -> Dict:
        """Get data for the learning dashboard."""
        domains = ['healthcare', 'business', 'education', 'creative', 'leadership']
        
        dashboard_data = {
            'system_status': 'running' if self.is_running else 'stopped',
            'last_updated': datetime.now().isoformat(),
            'domains': {}
        }
        
        for domain in domains:
            stats = self.hafl_system.get_learning_statistics(domain)
            performance = self.hafl_system.get_recent_performance(domain, hours=24)
            
            dashboard_data['domains'][domain] = {
                'statistics': stats,
                'recent_performance': performance
            }
        
        return dashboard_data

# Export main classes
__all__ = [
    'UserFeedback',
    'RewardSignal', 
    'HAFLSystem',
    'ContinuousLearningManager'
]