import { useState, useCallback, useRef } from 'react';
import { voiceAPI, Domain } from '../lib/voice-api';

export interface VoiceState {
  isEnabled: boolean;
  isPlaying: boolean;
  isLoading: boolean;
  error: string | null;
  currentDomain: Domain;
}

export function useVoice(initialDomain: Domain = 'universal') {
  const [state, setState] = useState<VoiceState>({
    isEnabled: true,
    isPlaying: false,
    isLoading: false,
    error: null,
    currentDomain: initialDomain,
  });

  const audioRef = useRef<HTMLAudioElement | null>(null);

  const playAudio = useCallback(async (audioUrl: string) => {
    try {
      setState(prev => ({ ...prev, isPlaying: true, error: null }));
      
      if (audioRef.current) {
        audioRef.current.pause();
      }

      audioRef.current = new Audio(audioUrl);
      audioRef.current.onended = () => {
        setState(prev => ({ ...prev, isPlaying: false }));
      };
      audioRef.current.onerror = () => {
        setState(prev => ({ 
          ...prev, 
          isPlaying: false, 
          error: 'Failed to play audio' 
        }));
      };

      await audioRef.current.play();
    } catch (error) {
      setState(prev => ({ 
        ...prev, 
        isPlaying: false, 
        error: `Audio playback failed: ${error}` 
      }));
    }
  }, []);

  const speak = useCallback(async (text: string) => {
    if (!state.isEnabled) return;

    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      const response = await voiceAPI.synthesizeSpeech(text, state.currentDomain);
      
      if (response.success && response.audio_url) {
        await playAudio(response.audio_url);
      } else {
        setState(prev => ({ 
          ...prev, 
          error: response.error || 'Failed to synthesize speech' 
        }));
      }
    } catch (error) {
      setState(prev => ({ 
        ...prev, 
        error: `Voice synthesis failed: ${error}` 
      }));
    } finally {
      setState(prev => ({ ...prev, isLoading: false }));
    }
  }, [state.isEnabled, state.currentDomain, playAudio]);

  const chatWithVoice = useCallback(async (message: string) => {
    if (!state.isEnabled) return null;

    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      const response = await voiceAPI.chatWithVoice(message, state.currentDomain);
      
      if (response.success) {
        if (response.audio_url) {
          await playAudio(response.audio_url);
        }
        return response.text_response || null;
      } else {
        setState(prev => ({ 
          ...prev, 
          error: response.error || 'Failed to get voice response' 
        }));
        return null;
      }
    } catch (error) {
      setState(prev => ({ 
        ...prev, 
        error: `Voice chat failed: ${error}` 
      }));
      return null;
    } finally {
      setState(prev => ({ ...prev, isLoading: false }));
    }
  }, [state.isEnabled, state.currentDomain, playAudio]);

  const toggleVoice = useCallback(() => {
    setState(prev => ({ ...prev, isEnabled: !prev.isEnabled }));
  }, []);

  const setDomain = useCallback((domain: Domain) => {
    setState(prev => ({ ...prev, currentDomain: domain }));
  }, []);

  const stopAudio = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      setState(prev => ({ ...prev, isPlaying: false }));
    }
  }, []);

  return {
    ...state,
    speak,
    chatWithVoice,
    toggleVoice,
    setDomain,
    stopAudio,
  };
} 