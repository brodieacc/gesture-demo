'use client';

import { useRef, useState, useCallback, useEffect } from 'react';
import { useHandTracking } from '@/hooks/useHandTracking';
import { useAudio } from '@/hooks/useAudio';
import { HDCGestureRecognizer, type Landmark, type PredictionResult } from '@/lib/hdc';

type AppMode = 'recognition' | 'training' | 'naming';

interface TrainingState {
  gestureName: string;
  examplesCollected: number;
  requiredExamples: number;
}

const REQUIRED_EXAMPLES = 5;

export default function GestureDemo() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const recognizerRef = useRef<HDCGestureRecognizer | null>(null);
  const lastPredictionRef = useRef<string | null>(null);

  const [mode, setMode] = useState<AppMode>('recognition');
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [trainingState, setTrainingState] = useState<TrainingState>({
    gestureName: '',
    examplesCollected: 0,
    requiredExamples: REQUIRED_EXAMPLES,
  });
  const [gestureNames, setGestureNames] = useState<string[]>([]);
  const [statusMessage, setStatusMessage] = useState<string>('');
  const [currentLandmarks, setCurrentLandmarks] = useState<Landmark[] | null>(null);

  const { playSound } = useAudio();

  // Initialize recognizer
  useEffect(() => {
    recognizerRef.current = new HDCGestureRecognizer(10000, 16, 0.25);
  }, []);

  // Show status message temporarily
  const showStatus = useCallback((message: string) => {
    setStatusMessage(message);
    setTimeout(() => setStatusMessage(''), 2500);
  }, []);

  // Handle frame updates from hand tracking
  const handleFrame = useCallback((landmarks: Landmark[] | null) => {
    setCurrentLandmarks(landmarks);

    if (!recognizerRef.current || !landmarks) {
      if (mode === 'recognition') {
        setPrediction(null);
        lastPredictionRef.current = null;
      }
      return;
    }

    const hv = recognizerRef.current.encodeLandmarks(landmarks);

    if (mode === 'recognition') {
      const result = recognizerRef.current.predict(hv);
      setPrediction(result);

      // Play sound on new recognition (not on every frame)
      if (result.prediction && result.prediction !== lastPredictionRef.current) {
        playSound('recognize');
        lastPredictionRef.current = result.prediction;
      } else if (!result.prediction) {
        lastPredictionRef.current = null;
      }
    }
  }, [mode, playSound]);

  const { isLoading, isReady, error, landmarks } = useHandTracking(
    videoRef,
    canvasRef,
    { onFrame: handleFrame }
  );

  // Capture training example
  const captureExample = useCallback(() => {
    if (mode !== 'training' || !currentLandmarks || !recognizerRef.current) {
      return;
    }

    const hv = recognizerRef.current.encodeLandmarks(currentLandmarks);
    const count = recognizerRef.current.addExample(trainingState.gestureName, hv);

    playSound('capture');

    setTrainingState(prev => ({
      ...prev,
      examplesCollected: count,
    }));

    if (count >= REQUIRED_EXAMPLES) {
      setTimeout(() => playSound('complete'), 200);
      showStatus(`Learned: ${trainingState.gestureName}`);
      setGestureNames(recognizerRef.current.getClassNames());
      setMode('recognition');
      setTrainingState({
        gestureName: '',
        examplesCollected: 0,
        requiredExamples: REQUIRED_EXAMPLES,
      });
    }
  }, [mode, currentLandmarks, trainingState.gestureName, showStatus, playSound]);

  // Start training mode
  const startTraining = useCallback(() => {
    playSound('click');
    setMode('naming');
    setTimeout(() => inputRef.current?.focus(), 100);
  }, [playSound]);

  // Confirm gesture name and start capturing
  const confirmGestureName = useCallback(() => {
    const name = inputRef.current?.value.trim().toUpperCase();
    if (!name) return;

    playSound('click');
    setTrainingState({
      gestureName: name,
      examplesCollected: 0,
      requiredExamples: REQUIRED_EXAMPLES,
    });
    setMode('training');
  }, [playSound]);

  // Cancel training
  const cancelTraining = useCallback(() => {
    playSound('click');
    setMode('recognition');
    setTrainingState({
      gestureName: '',
      examplesCollected: 0,
      requiredExamples: REQUIRED_EXAMPLES,
    });
  }, [playSound]);

  // Clear all gestures
  const clearAll = useCallback(() => {
    if (!recognizerRef.current) return;
    playSound('click');
    recognizerRef.current.clearAll();
    setGestureNames([]);
    setPrediction(null);
    showStatus('All gestures cleared');
  }, [showStatus, playSound]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if typing in input
      if (e.target instanceof HTMLInputElement) {
        if (e.key === 'Enter') {
          e.preventDefault();
          confirmGestureName();
        } else if (e.key === 'Escape') {
          cancelTraining();
        }
        return;
      }

      switch (e.key.toLowerCase()) {
        case 't':
          if (mode === 'recognition') startTraining();
          break;
        case 'r':
          if (mode === 'training' || mode === 'naming') cancelTraining();
          break;
        case ' ':
          if (mode === 'training') {
            e.preventDefault();
            captureExample();
          }
          break;
        case 'c':
          if (mode === 'recognition') clearAll();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [mode, startTraining, cancelTraining, captureExample, clearAll, confirmGestureName]);

  // Render similarity bars
  const renderSimilarityBars = () => {
    if (gestureNames.length === 0) {
      return (
        <div className="text-gray-400 text-sm">
          <p>No gestures learned yet.</p>
          <p className="mt-2">Press <kbd className="kbd">T</kbd> to train your first gesture.</p>
        </div>
      );
    }

    const similarities = prediction?.similarities ?? new Map();
    const sortedGestures = [...gestureNames].sort((a, b) => {
      return (similarities.get(b) ?? 0) - (similarities.get(a) ?? 0);
    });

    return (
      <div className="space-y-3">
        <h3 className="text-sm text-gray-400 font-medium uppercase tracking-wider">Similarities</h3>
        {sortedGestures.map(name => {
          const sim = similarities.get(name) ?? 0;
          const isTop = name === prediction?.prediction;
          const percentage = Math.max(0, Math.min(100, sim * 100));

          return (
            <div key={name} className="space-y-1">
              <div className="flex justify-between items-center">
                <span className={`font-mono font-medium ${isTop ? 'text-emerald-400' : 'text-white'}`}>
                  {name}
                </span>
                <span className={`font-mono text-sm ${isTop ? 'text-emerald-400' : 'text-gray-400'}`}>
                  {sim.toFixed(2)}
                </span>
              </div>
              <div className="h-6 bg-gray-700/50 rounded-md overflow-hidden">
                <div
                  className={`h-full transition-all duration-150 ${
                    isTop ? 'bg-emerald-500' : 'bg-blue-500'
                  }`}
                  style={{ width: `${percentage}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  // Render training progress
  const renderTrainingProgress = () => {
    const progress = (trainingState.examplesCollected / trainingState.requiredExamples) * 100;

    return (
      <div className="space-y-4">
        <div>
          <div className="flex justify-between items-center mb-2">
            <span className="text-white">Examples captured</span>
            <span className="text-blue-400 font-mono font-bold">
              {trainingState.examplesCollected} / {trainingState.requiredExamples}
            </span>
          </div>
          <div className="h-8 bg-gray-700/50 rounded-lg overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-blue-500 to-emerald-500 transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        <div className="flex justify-center gap-2">
          {Array.from({ length: REQUIRED_EXAMPLES }).map((_, i) => (
            <div
              key={i}
              className={`w-3 h-3 rounded-full transition-all duration-300 ${
                i < trainingState.examplesCollected
                  ? 'bg-emerald-500 scale-110'
                  : 'bg-gray-600'
              }`}
            />
          ))}
        </div>

        {currentLandmarks ? (
          <p className="text-emerald-400 text-center">
            Show gesture and press <kbd className="kbd">SPACE</kbd> to capture
          </p>
        ) : (
          <p className="text-yellow-400 text-center">
            Position your hand in the camera
          </p>
        )}

        <button
          onClick={cancelTraining}
          className="w-full py-2 px-4 bg-gray-700 hover:bg-gray-600 rounded-lg text-gray-300 transition-colors"
        >
          Cancel Training
        </button>
      </div>
    );
  };

  // Render naming dialog
  const renderNamingDialog = () => {
    return (
      <div className="space-y-4">
        <h3 className="text-white font-medium">Name your gesture</h3>
        <input
          ref={inputRef}
          type="text"
          placeholder="e.g., THUMBS_UP"
          className="w-full px-4 py-3 bg-gray-700/50 border border-gray-600 rounded-lg text-white font-mono placeholder-gray-500 focus:outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
          onKeyDown={(e) => {
            if (e.key === 'Enter') confirmGestureName();
            if (e.key === 'Escape') cancelTraining();
          }}
        />
        <div className="flex gap-2">
          <button
            onClick={confirmGestureName}
            className="flex-1 py-2 px-4 bg-blue-500 hover:bg-blue-600 rounded-lg text-white font-medium transition-colors"
          >
            Start Training
          </button>
          <button
            onClick={cancelTraining}
            className="py-2 px-4 bg-gray-700 hover:bg-gray-600 rounded-lg text-gray-300 transition-colors"
          >
            Cancel
          </button>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-4">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">
              <span className="text-blue-400">Few-Shot</span> Gesture Recognition
            </h1>
            <p className="text-gray-500 text-sm font-mono">
              Powered by Hyperdimensional Computing
            </p>
          </div>
          <div className="flex items-center gap-4">
            <div className={`px-3 py-1 rounded-full text-sm font-mono font-medium ${
              mode === 'recognition'
                ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                : 'bg-blue-500/20 text-blue-400 border border-blue-500/30'
            }`}>
              {mode === 'recognition' ? 'RECOGNITION' :
               mode === 'naming' ? 'NAMING' :
               `TRAINING: ${trainingState.gestureName}`}
            </div>
            <div className="text-gray-500 text-sm font-mono">
              <span className="text-blue-400">{gestureNames.length}</span> gesture{gestureNames.length !== 1 ? 's' : ''}
            </div>
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Camera feed */}
          <div className="lg:col-span-2">
            <div className="relative bg-gray-800 rounded-2xl overflow-hidden aspect-video border border-gray-700">
              {/* Hidden video element */}
              <video
                ref={videoRef}
                className="hidden"
                playsInline
                muted
              />

              {/* Canvas with overlay */}
              <canvas
                ref={canvasRef}
                className="w-full h-full object-cover"
              />

              {/* Loading overlay */}
              {isLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80">
                  <div className="text-center">
                    <div className="w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
                    <p className="text-gray-300 font-mono text-sm">Initializing camera...</p>
                  </div>
                </div>
              )}

              {/* Error overlay */}
              {error && (
                <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80">
                  <div className="text-center text-red-400">
                    <p className="text-lg font-medium mb-2">Camera Error</p>
                    <p className="text-sm font-mono">{error}</p>
                  </div>
                </div>
              )}

              {/* No hand detected indicator */}
              {isReady && !landmarks && mode === 'recognition' && (
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                  <div className="text-center text-gray-400 bg-gray-900/60 px-6 py-4 rounded-xl border border-gray-700">
                    <p className="text-lg">Show your hand</p>
                  </div>
                </div>
              )}

              {/* Status message */}
              {statusMessage && (
                <div className="absolute bottom-4 left-1/2 -translate-x-1/2 bg-emerald-500 text-white px-6 py-3 rounded-lg font-medium animate-pulse">
                  {statusMessage}
                </div>
              )}

              {/* Prediction overlay */}
              {mode === 'recognition' && prediction?.prediction && (
                <div className="absolute top-4 left-4 bg-gray-900/80 backdrop-blur px-4 py-2 rounded-lg border border-emerald-500/30">
                  <p className="text-emerald-400 font-mono font-bold text-xl">{prediction.prediction}</p>
                  <p className="text-gray-400 text-sm font-mono">
                    conf: {(prediction.confidence * 100).toFixed(0)}%
                  </p>
                </div>
              )}

              {/* Dimension indicator */}
              <div className="absolute bottom-4 right-4 text-gray-500 text-xs font-mono">
                D=10,000
              </div>
            </div>

            {/* Controls */}
            <div className="mt-4 flex flex-wrap gap-3 justify-center">
              {mode === 'recognition' ? (
                <>
                  <button
                    onClick={startTraining}
                    className="px-6 py-3 bg-blue-500 hover:bg-blue-600 rounded-lg font-medium transition-colors"
                  >
                    <kbd className="kbd mr-2">T</kbd>
                    Train New Gesture
                  </button>
                  <button
                    onClick={clearAll}
                    disabled={gestureNames.length === 0}
                    className="px-6 py-3 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg font-medium transition-colors"
                  >
                    <kbd className="kbd mr-2">C</kbd>
                    Clear All
                  </button>
                </>
              ) : mode === 'training' ? (
                <button
                  onClick={captureExample}
                  disabled={!currentLandmarks}
                  className="px-8 py-4 bg-emerald-500 hover:bg-emerald-600 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg font-bold text-lg transition-colors"
                >
                  <kbd className="kbd mr-2">SPACE</kbd>
                  Capture Example
                </button>
              ) : null}
            </div>
          </div>

          {/* Sidebar */}
          <div className="lg:col-span-1">
            <div className="bg-gray-800/50 rounded-2xl p-6 border border-gray-700">
              {mode === 'recognition' && renderSimilarityBars()}
              {mode === 'training' && renderTrainingProgress()}
              {mode === 'naming' && renderNamingDialog()}
            </div>

            {/* Info card */}
            <div className="mt-4 bg-gray-800/30 rounded-xl p-5 border border-gray-800">
              <h4 className="font-semibold text-white mb-4 flex items-center gap-2 text-lg">
                <span className="text-blue-400 font-mono text-sm bg-blue-500/10 px-2 py-0.5 rounded">HDC</span>
                How it works
              </h4>
              <ul className="space-y-3 font-mono text-sm">
                <li className="flex items-start gap-3">
                  <span className="text-blue-400 font-bold">1.</span>
                  <span className="text-gray-300">Train gestures with {REQUIRED_EXAMPLES} examples</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-blue-400 font-bold">2.</span>
                  <span className="text-gray-300">No neural network training</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-blue-400 font-bold">3.</span>
                  <span className="text-gray-300">Add new gestures incrementally</span>
                </li>
                <li className="flex items-start gap-3">
                  <span className="text-blue-400 font-bold">4.</span>
                  <span className="text-gray-300">Instant vector similarity matching</span>
                </li>
              </ul>
            </div>

            {/* Technical stats */}
            <div className="mt-4 grid grid-cols-2 gap-3">
              <div className="bg-gray-800/30 rounded-lg p-4 border border-gray-800">
                <p className="text-gray-500 text-xs uppercase tracking-wider mb-1">Dimensions</p>
                <p className="text-white font-mono font-bold text-xl">10,000</p>
              </div>
              <div className="bg-gray-800/30 rounded-lg p-4 border border-gray-800">
                <p className="text-gray-500 text-xs uppercase tracking-wider mb-1">Features</p>
                <p className="text-white font-mono font-bold text-xl">48</p>
              </div>
            </div>
          </div>
        </div>
      </main>

      {/* Explanation Section */}
      <section className="max-w-4xl mx-auto px-6 py-12 mt-8 border-t border-gray-800">
        <div className="space-y-8">
          {/* What You're Seeing */}
          <div>
            <h2 className="text-2xl font-bold text-white mb-4">What You&apos;re Seeing</h2>
            <p className="text-gray-300 text-lg leading-relaxed">
              This demo recognizes hand gestures using{' '}
              <strong className="text-white">Hyperdimensional Computing (HDC)</strong>—a
              brain-inspired approach to machine learning that replaces neural network training with
              algebraic operations on high-dimensional vectors.
            </p>
          </div>

          {/* How it works */}
          <div>
            <h3 className="text-xl font-semibold text-white mb-4">How it works:</h3>
            <ul className="space-y-3 text-gray-300">
              <li className="flex items-start gap-3">
                <span className="text-blue-400 mt-1">•</span>
                <span>Your hand landmarks (48 features) are encoded into a 10,000-dimensional hypervector</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-400 mt-1">•</span>
                <span>Each gesture class is represented by a single prototype vector—the &quot;bundle&quot; of its examples</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-400 mt-1">•</span>
                <span>Classification computes similarity between your current pose and all prototypes</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="text-blue-400 mt-1">•</span>
                <span>Adding a new gesture just means creating a new prototype—no retraining required</span>
              </li>
            </ul>
          </div>

          {/* Comparison Table */}
          <div>
            <h3 className="text-xl font-semibold text-white mb-4">What&apos;s different from traditional ML:</h3>
            <div className="overflow-hidden rounded-lg border border-gray-700">
              <table className="w-full">
                <thead>
                  <tr className="bg-gray-800">
                    <th className="px-4 py-3 text-left text-gray-400 font-medium"></th>
                    <th className="px-4 py-3 text-left text-gray-400 font-medium">Traditional ML</th>
                    <th className="px-4 py-3 text-left text-blue-400 font-medium">HDC</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-700">
                  <tr className="bg-gray-800/50">
                    <td className="px-4 py-3 text-gray-400">Training</td>
                    <td className="px-4 py-3 text-gray-300">Gradient descent (slow)</td>
                    <td className="px-4 py-3 text-gray-300">Bundling (instant)</td>
                  </tr>
                  <tr className="bg-gray-800/30">
                    <td className="px-4 py-3 text-gray-400">Add new class</td>
                    <td className="px-4 py-3 text-gray-300">Retrain model</td>
                    <td className="px-4 py-3 text-gray-300">Add prototype</td>
                  </tr>
                  <tr className="bg-gray-800/50">
                    <td className="px-4 py-3 text-gray-400">Examples needed</td>
                    <td className="px-4 py-3 text-gray-300">50-100+</td>
                    <td className="px-4 py-3 text-gray-300 font-mono text-blue-400">3-5</td>
                  </tr>
                  <tr className="bg-gray-800/30">
                    <td className="px-4 py-3 text-gray-400">Compute</td>
                    <td className="px-4 py-3 text-gray-300">GPU recommended</td>
                    <td className="px-4 py-3 text-gray-300">CPU only</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Bigger Picture */}
          <div>
            <p className="text-gray-300 text-lg leading-relaxed">
              <strong className="text-white">The bigger picture:</strong>{' '}
              This demo shows classification, but HDC&apos;s algebraic framework extends to compositional
              representations—encoding structure, sequences, and relations. That&apos;s the foundation
              for our work on prosthetics, robotics, and edge AI.
            </p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-gray-800 px-6 py-4 mt-8">
        <div className="max-w-7xl mx-auto text-center text-gray-600 text-sm">
          <p className="font-mono">
            Hyperdimensional Computing Demo by{' '}
            <span className="text-blue-400">Twistient</span>
          </p>
        </div>
      </footer>

      {/* Global styles for kbd */}
      <style jsx global>{`
        .kbd {
          display: inline-block;
          padding: 0.15em 0.5em;
          font-size: 0.8em;
          font-family: var(--font-mono), ui-monospace, monospace;
          background: rgba(59, 130, 246, 0.15);
          border-radius: 4px;
          border: 1px solid rgba(59, 130, 246, 0.3);
          color: rgb(147, 197, 253);
        }
      `}</style>
    </div>
  );
}
