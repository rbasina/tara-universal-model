// src/hooks/useVoice.ts
// TARA Voice React Hook for tara-ai-companion frontend integration

import { useState, useCallback, useRef, useEffect } from 'react';
import { taraVoiceAPI, Domain, VoiceResponse, ChatVoiceResponse } from '../lib/voice-api';

export interface UseVoiceReturn {
  // State
  isEnabled: boolean;
  isPlaying: boolean;
  isLoading: boolean;
  error: string | null;
  currentDomain: Domain;
  isServerAvailable: boolean;

  // Actions
  speak: (text: string) => Promise<boolean>;
  chatWithVoice: (message: string) => Promise<string | null>;
  toggleVoice: () => void;
  setDomain: (domain: Domain) => void;
  stopAudio: () => void;
  clearError: () => void;
  checkServerStatus: () => Promise<boolean>;
}

export function useVoice(initialDomain: Domain = 'universal'): UseVoiceReturn {
  // State
  const [isEnabled, setIsEnabled] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentDomain, setCurrentDomain] = useState<Domain>(initialDomain);
  const [isServerAvailable, setIsServerAvailable] = useState(false);

  // Refs
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Check server status on mount and when enabled
  useEffect(() => {
    if (isEnabled) {
      checkServerStatus();
    }
  }, [isEnabled]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopAudio();
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  /**
   * Check if voice server is available
   */
  const checkServerStatus = useCallback(async (): Promise<boolean> => {
    try {
      const available = await taraVoiceAPI.isVoiceServerAvailable();
      setIsServerAvailable(available);
      
      if (!available && isEnabled) {
        setError('Voice server is not available. Please ensure TARA voice server is running.');
      } else if (available && error?.includes('server')) {
        setError(null);
      }
      
      return available;
    } catch (err) {
      setIsServerAvailable(false);
      if (isEnabled) {
        setError('Failed to connect to voice server');
      }
      return false;
    }
  }, [isEnabled, error]);

  /**
   * Play audio from URL
   */
  const playAudio = useCallback(async (audioUrl: string): Promise<boolean> => {
    return new Promise((resolve) => {
      try {
        // Stop any currently playing audio
        stopAudio();

        // Create new audio element
        const audio = new Audio(audioUrl);
        audioRef.current = audio;

        // Set up event listeners
        audio.onloadstart = () => setIsLoading(true);
        audio.oncanplay = () => setIsLoading(false);
        
        audio.onplay = () => {
          setIsPlaying(true);
          setIsLoading(false);
        };

        audio.onended = () => {
          setIsPlaying(false);
          setIsLoading(false);
          resolve(true);
        };

        audio.onerror = (e) => {
          console.error('Audio playback error:', e);
          setError('Failed to play audio');
          setIsPlaying(false);
          setIsLoading(false);
          resolve(false);
        };

        // Start playback
        audio.play().catch((err) => {
          console.error('Audio play error:', err);
          setError('Failed to start audio playback');
          setIsPlaying(false);
          setIsLoading(false);
          resolve(false);
        });

      } catch (err) {
        console.error('Audio setup error:', err);
        setError('Failed to setup audio playback');
        setIsLoading(false);
        resolve(false);
      }
    });
  }, []);

  /**
   * Stop currently playing audio
   */
  const stopAudio = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      audioRef.current = null;
    }
    setIsPlaying(false);
    setIsLoading(false);
  }, []);

  /**
   * Synthesize and speak text
   */
  const speak = useCallback(async (text: string): Promise<boolean> => {
    if (!isEnabled) {
      console.warn('Voice is disabled');
      return false;
    }

    if (!text.trim()) {
      console.warn('No text to speak');
      return false;
    }

    // Check server availability
    const serverAvailable = await checkServerStatus();
    if (!serverAvailable) {
      return false;
    }

    try {
      setIsLoading(true);
      setError(null);

      // Cancel any ongoing request
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      abortControllerRef.current = new AbortController();

      const response: VoiceResponse = await taraVoiceAPI.synthesizeSpeech(text, currentDomain);

      if (response.success && response.audio_url) {
        const playbackSuccess = await playAudio(response.audio_url);
        return playbackSuccess;
      } else {
        setError(response.error || 'Failed to synthesize speech');
        return false;
      }
    } catch (err) {
      console.error('Speak error:', err);
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
      return false;
    } finally {
      setIsLoading(false);
    }
  }, [isEnabled, currentDomain, checkServerStatus, playAudio]);

  /**
   * Chat with voice response
   */
  const chatWithVoice = useCallback(async (message: string): Promise<string | null> => {
    if (!isEnabled) {
      console.warn('Voice is disabled');
      return null;
    }

    if (!message.trim()) {
      console.warn('No message to send');
      return null;
    }

    // Check server availability
    const serverAvailable = await checkServerStatus();
    if (!serverAvailable) {
      return null;
    }

    try {
      setIsLoading(true);
      setError(null);

      // Cancel any ongoing request
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      abortControllerRef.current = new AbortController();

      const response: ChatVoiceResponse = await taraVoiceAPI.chatWithVoice(message, currentDomain);

      if (response.success) {
        // Play audio if available
        if (response.audio_url) {
          // Don't await - let audio play in background
          playAudio(response.audio_url).catch(console.error);
        }

        return response.text_response || null;
      } else {
        setError(response.error || 'Failed to get voice response');
        return null;
      }
    } catch (err) {
      console.error('Chat with voice error:', err);
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
      return null;
    } finally {
      setIsLoading(false);
    }
  }, [isEnabled, currentDomain, checkServerStatus, playAudio]);

  /**
   * Toggle voice on/off
   */
  const toggleVoice = useCallback(() => {
    if (isEnabled) {
      // Disable voice
      stopAudio();
      setIsEnabled(false);
      setError(null);
    } else {
      // Enable voice
      setIsEnabled(true);
      // Check server status when enabling
      checkServerStatus();
    }
  }, [isEnabled, stopAudio, checkServerStatus]);

  /**
   * Change voice domain
   */
  const setDomain = useCallback((domain: Domain) => {
    setCurrentDomain(domain);
    // Clear any domain-specific errors
    if (error?.includes('domain')) {
      setError(null);
    }
  }, [error]);

  /**
   * Clear current error
   */
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    // State
    isEnabled,
    isPlaying,
    isLoading,
    error,
    currentDomain,
    isServerAvailable,

    // Actions
    speak,
    chatWithVoice,
    toggleVoice,
    setDomain,
    stopAudio,
    clearError,
    checkServerStatus,
  };
} 