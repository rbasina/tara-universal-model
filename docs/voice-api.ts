// src/lib/voice-api.ts
// TARA Voice API Client for tara-ai-companion frontend integration

export type Domain = 'healthcare' | 'business' | 'education' | 'creative' | 'leadership' | 'universal';

export interface VoiceResponse {
  success: boolean;
  audio_url?: string;
  processing_time?: number;
  error?: string;
}

export interface ChatVoiceResponse {
  success: boolean;
  text_response?: string;
  audio_url?: string;
  processing_time?: number;
  error?: string;
}

export interface TTSStatus {
  status: string;
  edge_tts_available: boolean;
  pyttsx3_available: boolean;
  domains: Domain[];
}

export class TARAVoiceAPI {
  private baseUrl: string;

  constructor(baseUrl: string = 'http://localhost:5000') {
    this.baseUrl = baseUrl;
  }

  /**
   * Synthesize speech from text using TARA's voice system
   */
  async synthesizeSpeech(text: string, domain: Domain = 'universal'): Promise<VoiceResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/tts/synthesize`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text,
          domain,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Voice synthesis error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error occurred',
      };
    }
  }

  /**
   * Chat with TARA and receive both text and voice response
   */
  async chatWithVoice(message: string, domain: Domain = 'universal'): Promise<ChatVoiceResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/chat_with_voice`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message,
          domain,
          voice_response: true,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Chat with voice error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error occurred',
      };
    }
  }

  /**
   * Get TTS system status
   */
  async getTTSStatus(): Promise<TTSStatus | null> {
    try {
      const response = await fetch(`${this.baseUrl}/tts/status`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('TTS status error:', error);
      return null;
    }
  }

  /**
   * Check if voice server is available
   */
  async isVoiceServerAvailable(): Promise<boolean> {
    const status = await this.getTTSStatus();
    return status !== null && status.status === 'ready';
  }
}

// Export singleton instance
export const taraVoiceAPI = new TARAVoiceAPI();

// Domain configurations with voice personalities
export const DOMAIN_CONFIGS = {
  healthcare: {
    name: 'Healthcare',
    description: 'Gentle, empathetic voice for medical contexts',
    voice: 'AriaNeural-gentle',
    color: 'text-green-600',
  },
  business: {
    name: 'Business',
    description: 'Professional, confident voice for business contexts',
    voice: 'JennyNeural-professional',
    color: 'text-blue-600',
  },
  education: {
    name: 'Education',
    description: 'Patient, encouraging voice for learning contexts',
    voice: 'AriaNeural-patient',
    color: 'text-purple-600',
  },
  creative: {
    name: 'Creative',
    description: 'Expressive, inspiring voice for creative contexts',
    voice: 'SoniaNeural-expressive',
    color: 'text-pink-600',
  },
  leadership: {
    name: 'Leadership',
    description: 'Authoritative, motivating voice for leadership contexts',
    voice: 'JennyNeural-authoritative',
    color: 'text-red-600',
  },
  universal: {
    name: 'Universal',
    description: 'Friendly, versatile voice for general conversations',
    voice: 'AriaNeural-friendly',
    color: 'text-gray-600',
  },
} as const; 