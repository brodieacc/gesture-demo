import { useCallback, useRef } from 'react';

type SoundType = 'capture' | 'complete' | 'recognize' | 'click';

/**
 * Custom hook for subtle audio feedback using Web Audio API
 * Creates pleasant, non-intrusive sounds for key interactions
 */
export function useAudio() {
  const audioContextRef = useRef<AudioContext | null>(null);

  const getAudioContext = useCallback(() => {
    if (!audioContextRef.current) {
      audioContextRef.current = new AudioContext();
    }
    return audioContextRef.current;
  }, []);

  const playSound = useCallback((type: SoundType) => {
    try {
      const ctx = getAudioContext();
      const now = ctx.currentTime;

      // Create oscillator and gain nodes
      const oscillator = ctx.createOscillator();
      const gainNode = ctx.createGain();

      oscillator.connect(gainNode);
      gainNode.connect(ctx.destination);

      // Configure based on sound type
      switch (type) {
        case 'capture':
          // Quick chirp - ascending tone
          oscillator.type = 'sine';
          oscillator.frequency.setValueAtTime(400, now);
          oscillator.frequency.exponentialRampToValueAtTime(800, now + 0.08);
          gainNode.gain.setValueAtTime(0.15, now);
          gainNode.gain.exponentialRampToValueAtTime(0.01, now + 0.1);
          oscillator.start(now);
          oscillator.stop(now + 0.1);
          break;

        case 'complete':
          // Success chord - two-note ascending
          oscillator.type = 'sine';
          oscillator.frequency.setValueAtTime(523.25, now); // C5
          oscillator.frequency.setValueAtTime(659.25, now + 0.12); // E5
          gainNode.gain.setValueAtTime(0.12, now);
          gainNode.gain.setValueAtTime(0.15, now + 0.12);
          gainNode.gain.exponentialRampToValueAtTime(0.01, now + 0.35);
          oscillator.start(now);
          oscillator.stop(now + 0.35);

          // Add second oscillator for richer sound
          const osc2 = ctx.createOscillator();
          const gain2 = ctx.createGain();
          osc2.connect(gain2);
          gain2.connect(ctx.destination);
          osc2.type = 'sine';
          osc2.frequency.setValueAtTime(659.25, now + 0.1); // E5
          osc2.frequency.setValueAtTime(783.99, now + 0.22); // G5
          gain2.gain.setValueAtTime(0, now);
          gain2.gain.setValueAtTime(0.1, now + 0.1);
          gain2.gain.exponentialRampToValueAtTime(0.01, now + 0.4);
          osc2.start(now + 0.1);
          osc2.stop(now + 0.4);
          break;

        case 'recognize':
          // Soft ping
          oscillator.type = 'sine';
          oscillator.frequency.setValueAtTime(880, now); // A5
          gainNode.gain.setValueAtTime(0.08, now);
          gainNode.gain.exponentialRampToValueAtTime(0.01, now + 0.15);
          oscillator.start(now);
          oscillator.stop(now + 0.15);
          break;

        case 'click':
          // Subtle tick
          oscillator.type = 'square';
          oscillator.frequency.setValueAtTime(600, now);
          gainNode.gain.setValueAtTime(0.05, now);
          gainNode.gain.exponentialRampToValueAtTime(0.01, now + 0.03);
          oscillator.start(now);
          oscillator.stop(now + 0.03);
          break;
      }
    } catch {
      // Silently fail if audio isn't available
    }
  }, [getAudioContext]);

  return { playSound };
}
