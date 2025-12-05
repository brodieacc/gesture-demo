/**
 * Hyperdimensional Computing (HDC) Gesture Encoder and Classifier
 *
 * This module implements few-shot learning for gesture recognition using
 * Vector Symbolic Architectures (VSA). Key improvements over naive encoding:
 *
 * 1. Position-invariant: Normalizes hand relative to wrist
 * 2. Scale-invariant: Scales by hand size
 * 3. Rich features: Encodes angles, distances, and relative positions
 *
 * The magic: Learn new gestures from just 3-5 examples, no gradients required.
 */

// MediaPipe hand landmark indices
const WRIST = 0;
const THUMB_CMC = 1;
const THUMB_MCP = 2;
const THUMB_IP = 3;
const THUMB_TIP = 4;
const INDEX_MCP = 5;
const INDEX_PIP = 6;
const INDEX_DIP = 7;
const INDEX_TIP = 8;
const MIDDLE_MCP = 9;
const MIDDLE_PIP = 10;
const MIDDLE_DIP = 11;
const MIDDLE_TIP = 12;
const RING_MCP = 13;
const RING_PIP = 14;
const RING_DIP = 15;
const RING_TIP = 16;
const PINKY_MCP = 17;
const PINKY_PIP = 18;
const PINKY_DIP = 19;
const PINKY_TIP = 20;

// Finger tip indices for easy iteration
const FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP];
const FINGER_MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP];
const FINGER_PIPS = [THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP];

export interface Landmark {
  x: number;
  y: number;
  z: number;
}

export interface GestureClass {
  name: string;
  prototype: Float32Array;
  exampleCount: number;
}

export interface PredictionResult {
  prediction: string | null;
  similarities: Map<string, number>;
  confidence: number;
}

/**
 * Seeded random number generator for reproducible item memory
 */
class SeededRandom {
  private seed: number;

  constructor(seed: number) {
    this.seed = seed;
  }

  next(): number {
    // Linear congruential generator
    this.seed = (this.seed * 1664525 + 1013904223) % 4294967296;
    return this.seed / 4294967296;
  }

  nextBipolar(): number {
    return this.next() < 0.5 ? -1 : 1;
  }
}

/**
 * HDC Gesture Recognizer with position-invariant encoding
 */
export class HDCGestureRecognizer {
  private dim: number;
  private numBins: number;
  private threshold: number;
  private featureHVs: Map<string, Float32Array>;
  private classes: Map<string, GestureClass>;
  private rng: SeededRandom;

  constructor(
    dim: number = 10000,
    numBins: number = 16,
    threshold: number = 0.25,
    seed: number = 42
  ) {
    this.dim = dim;
    this.numBins = numBins;
    this.threshold = threshold;
    this.featureHVs = new Map();
    this.classes = new Map();
    this.rng = new SeededRandom(seed);
  }

  /**
   * Get or create a random hypervector for a (feature, bin) pair
   */
  private getFeatureHV(featureIdx: number, binVal: number): Float32Array {
    const key = `${featureIdx}:${binVal}`;

    if (!this.featureHVs.has(key)) {
      // Generate deterministic random HV based on feature and bin
      const localRng = new SeededRandom(featureIdx * 1000 + binVal + 12345);
      const hv = new Float32Array(this.dim);
      for (let i = 0; i < this.dim; i++) {
        hv[i] = localRng.nextBipolar();
      }
      this.featureHVs.set(key, hv);
    }

    return this.featureHVs.get(key)!;
  }

  /**
   * Quantize a value to a bin index
   */
  private quantize(value: number, minVal: number, maxVal: number): number {
    const normalized = (value - minVal) / (maxVal - minVal + 1e-8);
    const clamped = Math.max(0, Math.min(1, normalized));
    const bin = Math.floor(clamped * this.numBins);
    return Math.min(bin, this.numBins - 1);
  }

  /**
   * Calculate distance between two landmarks
   */
  private distance(a: Landmark, b: Landmark): number {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    const dz = a.z - b.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  /**
   * Calculate angle between three landmarks (angle at middle point)
   */
  private angle(a: Landmark, b: Landmark, c: Landmark): number {
    const ba = { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
    const bc = { x: c.x - b.x, y: c.y - b.y, z: c.z - b.z };

    const dot = ba.x * bc.x + ba.y * bc.y + ba.z * bc.z;
    const magBa = Math.sqrt(ba.x * ba.x + ba.y * ba.y + ba.z * ba.z);
    const magBc = Math.sqrt(bc.x * bc.x + bc.y * bc.y + bc.z * bc.z);

    if (magBa < 1e-8 || magBc < 1e-8) return 0;

    const cosAngle = Math.max(-1, Math.min(1, dot / (magBa * magBc)));
    return Math.acos(cosAngle);
  }

  /**
   * Extract position-invariant features from hand landmarks
   */
  private extractFeatures(landmarks: Landmark[]): number[] {
    const features: number[] = [];

    // Reference point: wrist
    const wrist = landmarks[WRIST];

    // Reference scale: distance from wrist to middle finger MCP
    const middleMcp = landmarks[MIDDLE_MCP];
    const handSize = this.distance(wrist, middleMcp);

    if (handSize < 1e-6) {
      // Hand too small/not detected properly, return zeros
      return new Array(100).fill(0);
    }

    // Palm center (average of MCP joints)
    const palmCenter = {
      x: (landmarks[INDEX_MCP].x + landmarks[MIDDLE_MCP].x +
          landmarks[RING_MCP].x + landmarks[PINKY_MCP].x) / 4,
      y: (landmarks[INDEX_MCP].y + landmarks[MIDDLE_MCP].y +
          landmarks[RING_MCP].y + landmarks[PINKY_MCP].y) / 4,
      z: (landmarks[INDEX_MCP].z + landmarks[MIDDLE_MCP].z +
          landmarks[RING_MCP].z + landmarks[PINKY_MCP].z) / 4,
    };

    // Feature 1: Normalized fingertip distances from wrist (5 features)
    for (const tip of FINGERTIPS) {
      const dist = this.distance(landmarks[tip], wrist) / handSize;
      features.push(dist);
    }

    // Feature 2: Normalized fingertip distances from palm center (5 features)
    for (const tip of FINGERTIPS) {
      const dist = this.distance(landmarks[tip], palmCenter) / handSize;
      features.push(dist);
    }

    // Feature 3: Fingertip y-positions relative to wrist (are fingers up/down?) (5 features)
    for (const tip of FINGERTIPS) {
      const relY = (wrist.y - landmarks[tip].y) / handSize; // Positive = finger pointing up
      features.push(relY);
    }

    // Feature 4: Fingertip x-positions relative to palm center (spread) (5 features)
    for (const tip of FINGERTIPS) {
      const relX = (landmarks[tip].x - palmCenter.x) / handSize;
      features.push(relX);
    }

    // Feature 5: Finger curl angles (PIP joint angles) (5 features)
    // For each finger: angle at PIP between MCP and DIP/TIP
    const fingerJoints = [
      [THUMB_MCP, THUMB_IP, THUMB_TIP],
      [INDEX_MCP, INDEX_PIP, INDEX_TIP],
      [MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP],
      [RING_MCP, RING_PIP, RING_TIP],
      [PINKY_MCP, PINKY_PIP, PINKY_TIP],
    ];

    for (const [mcp, pip, tip] of fingerJoints) {
      const ang = this.angle(landmarks[mcp], landmarks[pip], landmarks[tip]);
      features.push(ang / Math.PI); // Normalize to [0, 1]
    }

    // Feature 6: Inter-fingertip distances (10 features - all pairs)
    for (let i = 0; i < FINGERTIPS.length; i++) {
      for (let j = i + 1; j < FINGERTIPS.length; j++) {
        const dist = this.distance(
          landmarks[FINGERTIPS[i]],
          landmarks[FINGERTIPS[j]]
        ) / handSize;
        features.push(dist);
      }
    }

    // Feature 7: Thumb-to-finger distances (important for pinch/OK gestures) (4 features)
    for (let i = 1; i < FINGERTIPS.length; i++) {
      const dist = this.distance(
        landmarks[THUMB_TIP],
        landmarks[FINGERTIPS[i]]
      ) / handSize;
      features.push(dist);
    }

    // Feature 8: Fingertip z-positions (depth relative to wrist) (5 features)
    for (const tip of FINGERTIPS) {
      const relZ = (landmarks[tip].z - wrist.z) / handSize;
      features.push(relZ);
    }

    // Feature 9: MCP joint spread (how open is the hand) (4 features)
    for (let i = 0; i < FINGER_MCPS.length - 1; i++) {
      const dist = this.distance(
        landmarks[FINGER_MCPS[i]],
        landmarks[FINGER_MCPS[i + 1]]
      ) / handSize;
      features.push(dist);
    }

    return features;
  }

  /**
   * Encode hand landmarks into a hypervector
   */
  encodeLandmarks(landmarks: Landmark[]): Float32Array {
    const features = this.extractFeatures(landmarks);

    // Bundle all feature hypervectors
    const hv = new Float32Array(this.dim);

    for (let i = 0; i < features.length; i++) {
      // Different feature types have different ranges
      let minVal = 0;
      let maxVal = 3; // Most normalized features are in [0, 3]

      // Adjust ranges for specific feature types
      if (i >= 10 && i < 15) {
        // Relative y-positions can be negative
        minVal = -2;
        maxVal = 2;
      } else if (i >= 15 && i < 20) {
        // Relative x-positions
        minVal = -2;
        maxVal = 2;
      } else if (i >= 20 && i < 25) {
        // Angles normalized to [0, 1]
        minVal = 0;
        maxVal = 1;
      } else if (i >= 43 && i < 48) {
        // Z positions
        minVal = -1;
        maxVal = 1;
      }

      const bin = this.quantize(features[i], minVal, maxVal);
      const featureHV = this.getFeatureHV(i, bin);

      // Add to bundle
      for (let j = 0; j < this.dim; j++) {
        hv[j] += featureHV[j];
      }
    }

    // Binarize by taking sign
    for (let i = 0; i < this.dim; i++) {
      hv[i] = hv[i] >= 0 ? 1 : -1;
    }

    return hv;
  }

  /**
   * Add a training example to a gesture class
   */
  addExample(gestureName: string, hv: Float32Array): number {
    const name = gestureName.toUpperCase();

    if (!this.classes.has(name)) {
      this.classes.set(name, {
        name,
        prototype: new Float32Array(this.dim),
        exampleCount: 0,
      });
    }

    const gestureClass = this.classes.get(name)!;

    // Bundle: add example to prototype
    for (let i = 0; i < this.dim; i++) {
      gestureClass.prototype[i] += hv[i];
    }
    gestureClass.exampleCount++;

    return gestureClass.exampleCount;
  }

  /**
   * Compute cosine similarity between two vectors
   */
  private cosineSimilarity(a: Float32Array, b: Float32Array): number {
    let dot = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < this.dim; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    if (denom < 1e-8) return 0;

    return dot / denom;
  }

  /**
   * Predict the gesture class for a hand pose
   */
  predict(hv: Float32Array): PredictionResult {
    const similarities = new Map<string, number>();

    if (this.classes.size === 0) {
      return { prediction: null, similarities, confidence: 0 };
    }

    let bestClass: string | null = null;
    let bestSim = -Infinity;

    for (const [name, gestureClass] of this.classes) {
      const sim = this.cosineSimilarity(hv, gestureClass.prototype);
      similarities.set(name, sim);

      if (sim > bestSim) {
        bestSim = sim;
        bestClass = name;
      }
    }

    // Only return prediction if above threshold
    if (bestSim < this.threshold) {
      return { prediction: null, similarities, confidence: bestSim };
    }

    return { prediction: bestClass, similarities, confidence: bestSim };
  }

  /**
   * Get all gesture class names
   */
  getClassNames(): string[] {
    return Array.from(this.classes.keys());
  }

  /**
   * Get example count for a class
   */
  getExampleCount(name: string): number {
    return this.classes.get(name.toUpperCase())?.exampleCount ?? 0;
  }

  /**
   * Clear all learned gestures
   */
  clearAll(): void {
    this.classes.clear();
  }

  /**
   * Remove a specific gesture
   */
  removeGesture(name: string): boolean {
    return this.classes.delete(name.toUpperCase());
  }

  /**
   * Export learned gestures to JSON-serializable format
   */
  export(): object {
    const data: Record<string, { prototype: number[]; exampleCount: number }> = {};

    for (const [name, gestureClass] of this.classes) {
      data[name] = {
        prototype: Array.from(gestureClass.prototype),
        exampleCount: gestureClass.exampleCount,
      };
    }

    return {
      dim: this.dim,
      numBins: this.numBins,
      threshold: this.threshold,
      classes: data,
    };
  }

  /**
   * Import learned gestures from exported format
   */
  import(data: {
    dim: number;
    numBins: number;
    threshold: number;
    classes: Record<string, { prototype: number[]; exampleCount: number }>;
  }): void {
    this.dim = data.dim;
    this.numBins = data.numBins;
    this.threshold = data.threshold;
    this.classes.clear();

    for (const [name, classData] of Object.entries(data.classes)) {
      this.classes.set(name, {
        name,
        prototype: new Float32Array(classData.prototype),
        exampleCount: classData.exampleCount,
      });
    }
  }
}
