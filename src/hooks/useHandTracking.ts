import { useEffect, useRef, useState, useCallback } from 'react';
import type { Landmark } from '@/lib/hdc';

export interface HandTrackingState {
  isLoading: boolean;
  isReady: boolean;
  error: string | null;
  landmarks: Landmark[] | null;
  handedness: 'Left' | 'Right' | null;
}

interface UseHandTrackingOptions {
  onFrame?: (landmarks: Landmark[] | null) => void;
}

declare global {
  interface Window {
    Hands: typeof import('@mediapipe/hands').Hands;
    drawConnectors: typeof import('@mediapipe/drawing_utils').drawConnectors;
    drawLandmarks: typeof import('@mediapipe/drawing_utils').drawLandmarks;
    HAND_CONNECTIONS: typeof import('@mediapipe/hands').HAND_CONNECTIONS;
  }
}

export function useHandTracking(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
  options: UseHandTrackingOptions = {}
) {
  const [state, setState] = useState<HandTrackingState>({
    isLoading: true,
    isReady: false,
    error: null,
    landmarks: null,
    handedness: null,
  });

  const handsRef = useRef<import('@mediapipe/hands').Hands | null>(null);
  const animationRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const drawHand = useCallback((
    ctx: CanvasRenderingContext2D,
    landmarks: Landmark[],
    width: number,
    height: number
  ) => {
    // Hand connections (simplified)
    const connections = [
      [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
      [0, 5], [5, 6], [6, 7], [7, 8], // Index
      [0, 9], [9, 10], [10, 11], [11, 12], // Middle
      [0, 13], [13, 14], [14, 15], [15, 16], // Ring
      [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
      [5, 9], [9, 13], [13, 17], // Palm
    ];

    // Draw connections with cyan/teal gradient
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';

    for (const [start, end] of connections) {
      const startLm = landmarks[start];
      const endLm = landmarks[end];

      // Create gradient for each connection
      const gradient = ctx.createLinearGradient(
        startLm.x * width, startLm.y * height,
        endLm.x * width, endLm.y * height
      );
      gradient.addColorStop(0, '#00E5CC'); // cyan/teal
      gradient.addColorStop(1, '#00FFE5'); // lighter cyan

      ctx.strokeStyle = gradient;
      ctx.beginPath();
      ctx.moveTo(startLm.x * width, startLm.y * height);
      ctx.lineTo(endLm.x * width, endLm.y * height);
      ctx.stroke();
    }

    // Draw landmarks
    for (let i = 0; i < landmarks.length; i++) {
      const lm = landmarks[i];
      const x = lm.x * width;
      const y = lm.y * height;

      // Glow effect
      ctx.shadowColor = '#00E5CC';
      ctx.shadowBlur = 8;

      // Outer circle
      ctx.fillStyle = '#00E5CC';
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, 2 * Math.PI);
      ctx.fill();

      // Inner circle
      ctx.shadowBlur = 0;
      ctx.fillStyle = '#003D36';
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, 2 * Math.PI);
      ctx.fill();
    }

    // Reset shadow
    ctx.shadowBlur = 0;
  }, []);

  useEffect(() => {
    let mounted = true;

    const initializeHandTracking = async () => {
      try {
        // Dynamic import for MediaPipe
        const handsModule = await import('@mediapipe/hands');

        const hands = new handsModule.Hands({
          locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
          },
        });

        hands.setOptions({
          maxNumHands: 1,
          modelComplexity: 1,
          minDetectionConfidence: 0.7,
          minTrackingConfidence: 0.5,
        });

        hands.onResults((results) => {
          if (!mounted) return;

          const canvas = canvasRef.current;
          const video = videoRef.current;
          if (!canvas || !video) return;

          const ctx = canvas.getContext('2d');
          if (!ctx) return;

          // Clear canvas
          ctx.clearRect(0, 0, canvas.width, canvas.height);

          // Draw video frame (mirrored)
          ctx.save();
          ctx.scale(-1, 1);
          ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
          ctx.restore();

          if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
            const landmarks = results.multiHandLandmarks[0] as Landmark[];
            const handedness = results.multiHandedness?.[0]?.label as 'Left' | 'Right' || null;

            // Mirror x coordinates for display
            const mirroredLandmarks = landmarks.map(lm => ({
              x: 1 - lm.x,
              y: lm.y,
              z: lm.z,
            }));

            drawHand(ctx, mirroredLandmarks, canvas.width, canvas.height);

            setState(prev => ({
              ...prev,
              landmarks: mirroredLandmarks,
              handedness,
            }));

            options.onFrame?.(mirroredLandmarks);
          } else {
            setState(prev => ({
              ...prev,
              landmarks: null,
              handedness: null,
            }));

            options.onFrame?.(null);
          }
        });

        handsRef.current = hands;

        // Get webcam
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            facingMode: 'user',
          },
        });

        streamRef.current = stream;

        const video = videoRef.current;
        if (video) {
          video.srcObject = stream;
          await video.play();

          // Set canvas size
          const canvas = canvasRef.current;
          if (canvas) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
          }

          if (mounted) {
            setState(prev => ({
              ...prev,
              isLoading: false,
              isReady: true,
            }));
          }

          // Start processing frames
          const processFrame = async () => {
            if (!mounted || !handsRef.current || !video) return;

            try {
              await handsRef.current.send({ image: video });
            } catch {
              // Ignore frame processing errors
            }

            animationRef.current = requestAnimationFrame(processFrame);
          };

          processFrame();
        }
      } catch (error) {
        if (mounted) {
          setState(prev => ({
            ...prev,
            isLoading: false,
            error: error instanceof Error ? error.message : 'Failed to initialize hand tracking',
          }));
        }
      }
    };

    initializeHandTracking();

    return () => {
      mounted = false;

      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }

      if (handsRef.current) {
        handsRef.current.close();
      }

      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, [videoRef, canvasRef, options.onFrame, drawHand]);

  return state;
}
