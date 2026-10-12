/**
 * sherpa-onnx-node ships no types: the part of its API the transcription
 * engine uses (./sherpa), as its JSDoc in types.js describes it.
 */
declare module "sherpa-onnx-node" {
  export interface Waveform {
    samples: Float32Array;
    sampleRate: number;
  }

  export interface OfflineRecognizerConfig {
    featConfig: { sampleRate: number; featureDim: number };
    modelConfig: {
      qwen3Asr?: {
        convFrontend: string;
        encoder: string;
        decoder: string;
        tokenizer: string;
        hotwords?: string;
        maxNewTokens?: number;
      };
      tokens: string;
      numThreads: number;
      provider: "cpu";
      debug: number;
    };
  }

  export interface OfflineStream {
    acceptWaveform(waveform: Waveform): void;
  }

  export class OfflineRecognizer {
    static createAsync(config: OfflineRecognizerConfig): Promise<OfflineRecognizer>;
    createStream(): OfflineStream;
    decodeAsync(stream: OfflineStream): Promise<{ text: string }>;
  }

  export interface VadConfig {
    sileroVad: {
      model: string;
      threshold: number;
      minSpeechDuration: number;
      minSilenceDuration: number;
      maxSpeechDuration: number;
      windowSize: number;
    };
    sampleRate: number;
    numThreads: number;
    debug: number;
  }

  export class Vad {
    constructor(config: VadConfig, bufferSizeInSeconds: number);
    acceptWaveform(samples: Float32Array): void;
    isEmpty(): boolean;
    isDetected(): boolean;
    front(enableExternalBuffer?: boolean): { start: number; samples: Float32Array };
    pop(): void;
    flush(): void;
  }
}
