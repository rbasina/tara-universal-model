'use client';

import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  MessageCircle,
  Heart,
  Brain,
  Mic,
  MicOff,
  Send,
  Sparkles,
  Volume2,
  VolumeX,
  Video,
  VideoOff,
  Loader2
} from 'lucide-react';
import { useVoice } from '../hooks/useVoice';
import { Domain } from '../lib/voice-api';

interface Message {
  id: string;
  type: 'user' | 'tara';
  content: string;
  timestamp: Date;
  emotion?: string;
  category: 'emotional' | 'knowledge' | 'general';
  hasAudio?: boolean;
}

interface MeeTARAVoiceProps {
  className?: string;
  onEmotionDetected?: (emotion: string, intensity: number) => void;
  voiceEnabled?: boolean;
  facialRecognition?: boolean;
  domain?: Domain;
}

export default function MeeTARAVoice({
  className = "",
  onEmotionDetected,
  voiceEnabled = true,
  facialRecognition = false,
  domain = 'universal'
}: MeeTARAVoiceProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [isFacialEnabled, setIsFacialEnabled] = useState(facialRecognition);
  const [isTyping, setIsTyping] = useState(false);
  const [currentEmotion, setCurrentEmotion] = useState<string>('neutral');

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // TARA Voice Integration
  const voice = useVoice(domain);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Welcome message on mount with voice
  useEffect(() => {
    const welcomeMessage: Message = {
      id: 'welcome',
      type: 'tara',
      content: "Hi! I'm TARA, your AI companion. I'm here to listen, support, and help you with anything on your mind. How are you feeling today?",
      timestamp: new Date(),
      emotion: 'friendly',
      category: 'emotional',
      hasAudio: true
    };
    setMessages([welcomeMessage]);
    
    // Speak welcome message if voice is enabled
    if (voice.isEnabled) {
      setTimeout(() => {
        voice.speak(welcomeMessage.content);
      }, 1000);
    }
  }, []);

  const handleSendMessage = async () => {
    if (!inputText.trim()) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputText,
      timestamp: new Date(),
      category: 'general'
    };

    setMessages(prev => [...prev, userMessage]);
    const currentInput = inputText;
    setInputText('');
    setIsTyping(true);

    try {
      // Use TARA voice chat if enabled, otherwise generate response
      if (voice.isEnabled) {
        const voiceResponse = await voice.chatWithVoice(currentInput);
        
        if (voiceResponse) {
          const taraMessage: Message = {
            id: Date.now().toString(),
            type: 'tara',
            content: voiceResponse,
            timestamp: new Date(),
            emotion: detectEmotion(voiceResponse),
            category: categorizeMessage(voiceResponse),
            hasAudio: true
          };
          
          setMessages(prev => [...prev, taraMessage]);
          
          // Trigger emotion detection callback
          if (onEmotionDetected && taraMessage.emotion) {
            onEmotionDetected(taraMessage.emotion, 0.8);
          }
        }
      } else {
        // Fallback to local response generation
        setTimeout(() => {
          const taraResponse = generateTARAResponse(currentInput);
          setMessages(prev => [...prev, taraResponse]);
          
          // Trigger emotion detection callback
          if (onEmotionDetected && taraResponse.emotion) {
            onEmotionDetected(taraResponse.emotion, 0.8);
          }
        }, 1500);
      }
    } catch (error) {
      console.error('Error in chat:', error);
      // Fallback to local response
      const taraResponse = generateTARAResponse(currentInput);
      setMessages(prev => [...prev, taraResponse]);
    } finally {
      setIsTyping(false);
    }
  };

  const detectEmotion = (text: string): string => {
    const lowerText = text.toLowerCase();
    if (lowerText.includes('sorry') || lowerText.includes('understand')) return 'empathetic';
    if (lowerText.includes('wonderful') || lowerText.includes('great')) return 'joyful';
    if (lowerText.includes('help') || lowerText.includes('support')) return 'supportive';
    return 'friendly';
  };

  const categorizeMessage = (text: string): 'emotional' | 'knowledge' | 'general' => {
    const lowerText = text.toLowerCase();
    if (lowerText.includes('feel') || lowerText.includes('emotion')) return 'emotional';
    if (lowerText.includes('learn') || lowerText.includes('explain')) return 'knowledge';
    return 'general';
  };

  const generateTARAResponse = (userInput: string): Message => {
    const lowerInput = userInput.toLowerCase();
    let content = '';
    let emotion = 'neutral';
    let category: 'emotional' | 'knowledge' | 'general' = 'general';

    // Emotional responses
    if (lowerInput.includes('sad') || lowerInput.includes('upset') || lowerInput.includes('down')) {
      content = "I can sense you're going through a difficult time. It's completely normal to feel this way, and I want you to know that your feelings are valid. Would you like to talk about what's making you feel sad?";
      emotion = 'empathetic';
      category = 'emotional';
    } else if (lowerInput.includes('happy') || lowerInput.includes('excited') || lowerInput.includes('great')) {
      content = "That's wonderful! I can feel your positive energy, and it's contagious! I'm so happy to hear you're doing well. What's bringing you joy today?";
      emotion = 'joyful';
      category = 'emotional';
    } else if (lowerInput.includes('stressed') || lowerInput.includes('anxious') || lowerInput.includes('worried')) {
      content = "I understand you're feeling stressed right now. Take a deep breath with me. Stress is your mind's way of telling you something needs attention. Let's work through this together. What's the main thing on your mind?";
      emotion = 'supportive';
      category = 'emotional';
    } else if (lowerInput.includes('how') || lowerInput.includes('what') || lowerInput.includes('why') || lowerInput.includes('explain')) {
      content = "That's a great question! I love curious minds. Let me help you understand this better. Based on what you're asking, I can provide some insights and we can explore this topic together.";
      emotion = 'helpful';
      category = 'knowledge';
    } else {
      content = "I'm here to listen and support you. Every conversation with you helps me understand you better. Tell me what's on your mind - whether it's something you're curious about, how you're feeling, or just want to chat.";
      emotion = 'friendly';
      category = 'general';
    }

    const message: Message = {
      id: Date.now().toString(),
      type: 'tara',
      content,
      timestamp: new Date(),
      emotion,
      category
    };

    // Speak the response if voice is enabled
    if (voice.isEnabled) {
      setTimeout(() => {
        voice.speak(content);
      }, 500);
    }

    return message;
  };

  const handleVoiceToggle = () => {
    voice.toggleVoice();
  };

  const handleSpeakMessage = (message: Message) => {
    if (voice.isEnabled && message.type === 'tara') {
      voice.speak(message.content);
    }
  };

  const toggleFacial = () => {
    setIsFacialEnabled(!isFacialEnabled);
  };

  const getEmotionColor = (emotion?: string) => {
    const colors = {
      friendly: 'text-blue-500',
      empathetic: 'text-purple-500',
      joyful: 'text-yellow-500',
      supportive: 'text-green-500',
      helpful: 'text-indigo-500',
      neutral: 'text-gray-500'
    };
    return colors[emotion as keyof typeof colors] || colors.neutral;
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'emotional': return <Heart className="h-4 w-4" />;
      case 'knowledge': return <Brain className="h-4 w-4" />;
      default: return <MessageCircle className="h-4 w-4" />;
    }
  };

  return (
    <div className={`flex flex-col h-full bg-gradient-to-br from-slate-50 to-blue-50 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 bg-white/80 backdrop-blur-sm border-b border-slate-200">
        <div className="flex items-center space-x-3">
          <div className="relative">
            <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
              <Sparkles className="h-5 w-5 text-white" />
            </div>
            <div className={`absolute -bottom-1 -right-1 w-4 h-4 rounded-full border-2 border-white ${
              voice.isPlaying ? 'bg-green-500 animate-pulse' : 
              voice.isLoading ? 'bg-yellow-500 animate-spin' : 
              voice.isEnabled ? 'bg-blue-500' : 'bg-gray-400'
            }`}>
              {voice.isLoading && <Loader2 className="h-2 w-2 text-white animate-spin" />}
            </div>
          </div>
          <div>
            <h3 className="font-semibold text-slate-800">TARA</h3>
            <p className="text-xs text-slate-500 capitalize">
              {voice.currentDomain} • {voice.isEnabled ? 'Voice On' : 'Voice Off'}
            </p>
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center space-x-2">
          <button
            onClick={handleVoiceToggle}
            className={`p-2 rounded-lg transition-colors ${
              voice.isEnabled 
                ? 'bg-blue-100 text-blue-600 hover:bg-blue-200' 
                : 'bg-gray-100 text-gray-400 hover:bg-gray-200'
            }`}
            title={voice.isEnabled ? 'Disable Voice' : 'Enable Voice'}
          >
            {voice.isEnabled ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
          </button>
          
          <button
            onClick={toggleFacial}
            className={`p-2 rounded-lg transition-colors ${
              isFacialEnabled 
                ? 'bg-green-100 text-green-600 hover:bg-green-200' 
                : 'bg-gray-100 text-gray-400 hover:bg-gray-200'
            }`}
            title={isFacialEnabled ? 'Disable Facial Recognition' : 'Enable Facial Recognition'}
          >
            {isFacialEnabled ? <Video className="h-4 w-4" /> : <VideoOff className="h-4 w-4" />}
          </button>
        </div>
      </div>

      {/* Voice Status */}
      {voice.error && (
        <div className="mx-4 mt-2 p-2 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-sm text-red-600">Voice Error: {voice.error}</p>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        <AnimatePresence>
          {messages.map((message) => (
            <motion.div
              key={message.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`max-w-[80%] ${message.type === 'user' ? 'order-2' : 'order-1'}`}>
                <div
                  className={`p-3 rounded-2xl ${
                    message.type === 'user'
                      ? 'bg-blue-500 text-white'
                      : 'bg-white border border-slate-200 text-slate-800'
                  }`}
                >
                  <div className="flex items-start space-x-2">
                    {message.type === 'tara' && (
                      <div className={`mt-1 ${getEmotionColor(message.emotion)}`}>
                        {getCategoryIcon(message.category)}
                      </div>
                    )}
                    <div className="flex-1">
                      <p className="text-sm leading-relaxed">{message.content}</p>
                      <div className="flex items-center justify-between mt-2">
                        <span className="text-xs opacity-60">
                          {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </span>
                        {message.type === 'tara' && message.hasAudio && (
                          <button
                            onClick={() => handleSpeakMessage(message)}
                            className="text-xs opacity-60 hover:opacity-100 transition-opacity flex items-center space-x-1"
                            disabled={voice.isLoading || voice.isPlaying}
                          >
                            {voice.isLoading ? (
                              <Loader2 className="h-3 w-3 animate-spin" />
                            ) : (
                              <Volume2 className="h-3 w-3" />
                            )}
                            <span>Replay</span>
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {/* Typing Indicator */}
        {isTyping && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex justify-start"
          >
            <div className="bg-white border border-slate-200 rounded-2xl p-3">
              <div className="flex items-center space-x-2">
                <div className="text-blue-500">
                  <MessageCircle className="h-4 w-4" />
                </div>
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
                </div>
              </div>
            </div>
          </motion.div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-4 bg-white/80 backdrop-blur-sm border-t border-slate-200">
        <div className="flex items-center space-x-3">
          <div className="flex-1 relative">
            <input
              ref={inputRef}
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
              placeholder="Type your message to TARA..."
              className="w-full px-4 py-3 bg-slate-50 border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              disabled={voice.isLoading}
            />
            {voice.isLoading && (
              <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
                <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
              </div>
            )}
          </div>
          
          <button
            onClick={handleSendMessage}
            disabled={!inputText.trim() || voice.isLoading}
            className="p-3 bg-blue-500 text-white rounded-xl hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Send className="h-4 w-4" />
          </button>
        </div>
      </div>
    </div>
  );
} 