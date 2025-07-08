import { initLlama, LlamaContext } from './index'
import type {
  ContextParams,
  CompletionParams,
  CactusOAICompatibleMessage,
  NativeCompletionResult,
  EmbeddingParams,
  NativeEmbeddingResult,
} from './index'
import { Telemetry } from './telemetry'

interface CactusLMReturn {
  lm: CactusLM | null
  error: Error | null
}

export class CactusLM {
  private context: LlamaContext

  private constructor(context: LlamaContext) {
    this.context = context
  }

  static async init(
    params: ContextParams,
    onProgress?: (progress: number) => void,
  ): Promise<CactusLMReturn> {
    const configs = [
      params,
      { ...params, n_gpu_layers: 0 } 
    ];

    for (const config of configs) {
      try {
        const context = await initLlama(config, onProgress);
        return { lm: new CactusLM(context), error: null };
      } catch (e) {
        Telemetry.error(e as Error, {
          n_gpu_layers: config.n_gpu_layers ?? null,
          n_ctx: config.n_ctx ?? null,
          model: config.model ?? null,
        });
        if (configs.indexOf(config) === configs.length - 1) {
          return { lm: null, error: e as Error };
        }
      }
    }
    return { lm: null, error: new Error('Failed to initialize CactusLM') };
  }

  async completion(
    messages: CactusOAICompatibleMessage[],
    params: CompletionParams = {},
    callback?: (data: any) => void,
  ): Promise<NativeCompletionResult> {
    return await this.context.completion({ messages, ...params }, callback);
  }

  async embedding(
    text: string,
    params?: EmbeddingParams,
  ): Promise<NativeEmbeddingResult> {
    return this.context.embedding(text, params)
  }

  async rewind(): Promise<void> {
    // @ts-ignore
    return this.context?.rewind()
  }

  async release(): Promise<void> {
    return this.context.release()
  }
} 