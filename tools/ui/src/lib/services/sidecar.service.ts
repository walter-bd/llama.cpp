import {
	API_NATIVE_COMPLETION,
	API_SIDECAR,
	API_TOKENIZE
} from '$lib/constants/api-endpoints.constants';
import type {
	ApiNativeCompletionResponse,
	ApiSidecarLearningMode,
	ApiSidecarMutationResponse,
	ApiSidecarSequenceResponse,
	ApiSidecarStatusResponse
} from '$lib/types/api';
import { apiFetch, apiPost } from '$lib/utils/api-fetch';

interface TokenizeResponse {
	tokens: number[];
}

export interface SidecarObserveRequest {
	id: string;
	episode_id: string;
	observed_token_id?: number;
	observed_token?: string;
	observed_target?: string;
}

export interface SidecarObserveSequenceRequest {
	id: string;
	text: string;
	mode: ApiSidecarLearningMode;
	chunk_size?: number;
}

/** Thin client for the server's explicit sidecar lifecycle and observe API. */
export class SidecarService {
	static status(): Promise<ApiSidecarStatusResponse> {
		return apiFetch<ApiSidecarStatusResponse>(API_SIDECAR.STATUS);
	}

	static create(id: string, learning_enabled: boolean): Promise<Record<string, unknown>> {
		return apiPost(API_SIDECAR.CREATE, { id, learning_enabled });
	}

	static load(
		id: string,
		path: string,
		learning_enabled: boolean
	): Promise<Record<string, unknown>> {
		return apiPost(API_SIDECAR.LOAD, { id, learning_enabled, path });
	}

	static save(id: string, path?: string): Promise<Record<string, unknown>> {
		const body: { id: string; path?: string } = { id };
		const normalizedPath = path?.trim();
		if (normalizedPath) body.path = normalizedPath;
		return apiPost(API_SIDECAR.SAVE, body);
	}

	static reset(id: string): Promise<Record<string, unknown>> {
		return apiPost(API_SIDECAR.RESET, { id });
	}

	static unload(id: string): Promise<Record<string, unknown>> {
		return apiPost(API_SIDECAR.UNLOAD, { id });
	}

	static remove(id: string): Promise<Record<string, unknown>> {
		return apiFetch(`${API_SIDECAR.BASE}/${encodeURIComponent(id)}`, { method: 'DELETE' });
	}

	static setLearning(id: string, enabled: boolean): Promise<Record<string, unknown>> {
		return apiPost(API_SIDECAR.LEARNING, { enabled, id });
	}

	static tokenize(content: string): Promise<TokenizeResponse> {
		return apiPost<TokenizeResponse>(API_TOKENIZE, {
			add_special: false,
			content,
			parse_special: false
		});
	}

	static observe(request: SidecarObserveRequest): Promise<ApiSidecarMutationResponse> {
		return apiPost<ApiSidecarMutationResponse>(API_SIDECAR.OBSERVE, request);
	}

	static observeSequence(
		request: SidecarObserveSequenceRequest
	): Promise<ApiSidecarSequenceResponse> {
		return apiPost<ApiSidecarSequenceResponse>(API_SIDECAR.OBSERVE_SEQUENCE, request);
	}

	static async teachSingleToken(
		id: string,
		prompt: string,
		observedToken: string,
		observedTarget = observedToken
	): Promise<ApiSidecarMutationResponse> {
		const tokenized = await SidecarService.tokenize(observedToken);

		if (!Array.isArray(tokenized.tokens) || tokenized.tokens.length !== 1) {
			throw new Error(
				`Observed value must tokenize to exactly one token; received ${tokenized.tokens?.length ?? 0}.`
			);
		}

		const episode = await apiPost<ApiNativeCompletionResponse>(API_NATIVE_COMPLETION, {
			cache_prompt: false,
			n_predict: 1,
			prompt,
			sidecar_id: id,
			temperature: 0
		});

		if (!episode.sidecar_episode_id) {
			throw new Error('The server did not return a sidecar episode for this prompt.');
		}

		return SidecarService.observe({
			episode_id: episode.sidecar_episode_id,
			id,
			observed_target: observedTarget,
			observed_token_id: tokenized.tokens[0]
		});
	}
}
