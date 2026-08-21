import { browser } from '$app/environment';
import { SidecarService } from '$lib/services/sidecar.service';
import type {
	ApiSidecarLearningMode,
	ApiSidecarMutationResponse,
	ApiSidecarSequenceResponse,
	ApiSidecarStatus
} from '$lib/types/api';

const SELECTED_ID_KEY = 'llama-sidecar-selected-id';
const INTERNAL_PLASTICITY_KEY = 'llama-sidecar-internal-plasticity';

class SidecarStore {
	sidecars = $state<ApiSidecarStatus[]>([]);
	selectedId = $state('');
	enabled = $state(false);
	loading = $state(false);
	error = $state<string | null>(null);
	message = $state<string | null>(null);
	experimentalInternalPlasticity = $state(false);

	private initializePromise: Promise<void> | null = null;

	get selectedSidecar(): ApiSidecarStatus | undefined {
		return this.sidecars.find((sidecar) => sidecar.id === this.selectedId);
	}

	async initialize(): Promise<void> {
		if (!browser) return;

		if (this.initializePromise) return this.initializePromise;

		this.initializePromise = (async () => {
			this.selectedId = window.localStorage.getItem(SELECTED_ID_KEY) ?? '';
			this.experimentalInternalPlasticity =
				window.localStorage.getItem(INTERNAL_PLASTICITY_KEY) === 'true';
			await this.refresh();
		})();

		return this.initializePromise;
	}

	async refresh(): Promise<void> {
		try {
			const response = await SidecarService.status();

			this.enabled = response.enabled;
			this.sidecars = response.sidecars ?? [];
			this.error = null;

			if (!this.sidecars.some((sidecar) => sidecar.id === this.selectedId)) {
				this.select('');
			}
		} catch (error) {
			this.error = error instanceof Error ? error.message : String(error);
		}
	}

	select(id: string): void {
		this.selectedId = id;

		if (browser) {
			if (id) window.localStorage.setItem(SELECTED_ID_KEY, id);
			else window.localStorage.removeItem(SELECTED_ID_KEY);
		}
	}

	setExperimentalInternalPlasticity(enabled: boolean): void {
		this.experimentalInternalPlasticity = enabled;

		if (browser) {
			window.localStorage.setItem(INTERNAL_PLASTICITY_KEY, String(enabled));
		}
	}

	chatRequestParams(): Record<string, unknown> {
		const selected = this.selectedSidecar;

		if (!selected?.loaded) return {};

		return {
			sidecar_id: selected.id,
			...(this.experimentalInternalPlasticity ? { sidecar_debug_internal_plasticity: true } : {})
		};
	}

	async create(id: string, learningEnabled: boolean): Promise<void> {
		await this.action(() => SidecarService.create(id, learningEnabled), `Created ${id}.`);
		this.select(id);
		await this.refresh();
	}

	async load(id: string, path: string, learningEnabled: boolean): Promise<void> {
		await this.action(() => SidecarService.load(id, path, learningEnabled), `Loaded ${id}.`);
		this.select(id);
		await this.refresh();
	}

	async save(path: string): Promise<void> {
		const selected = this.selectedSidecar;

		if (!selected) throw new Error('Select a loaded sidecar first.');

		await this.action(
			() => SidecarService.save(selected.id, path),
			`Saved ${selected.id}.`
		);
		await this.refresh();
	}

	async reset(): Promise<void> {
		const selected = this.selectedSidecar;

		if (!selected) throw new Error('Select a loaded sidecar first.');

		await this.action(() => SidecarService.reset(selected.id), `Reset ${selected.id}.`);
		await this.refresh();
	}

	async unload(): Promise<void> {
		const selected = this.selectedSidecar;

		if (!selected) throw new Error('Select a loaded sidecar first.');

		await this.action(() => SidecarService.unload(selected.id), `Unloaded ${selected.id}.`);
		this.select('');
		await this.refresh();
	}

	async remove(): Promise<void> {
		const selected = this.selectedSidecar;

		if (!selected) throw new Error('Select a sidecar first.');

		await this.action(() => SidecarService.remove(selected.id), `Removed ${selected.id}.`);
		this.select('');
		await this.refresh();
	}

	async setLearning(enabled: boolean): Promise<void> {
		const selected = this.selectedSidecar;

		if (!selected) throw new Error('Select a loaded sidecar first.');

		await this.action(
			() => SidecarService.setLearning(selected.id, enabled),
			`Persistent learning ${enabled ? 'enabled' : 'disabled'}.`
		);
		await this.refresh();
	}

	async teach(prompt: string, observedToken: string): Promise<ApiSidecarMutationResponse> {
		const selected = this.selectedSidecar;

		if (!selected?.loaded) throw new Error('Select a loaded sidecar first.');

		this.loading = true;
		this.error = null;
		this.message = null;
		try {
			const result = await SidecarService.teachSingleToken(selected.id, prompt, observedToken);

			this.message = result.updated
				? `Observation accepted; state changed to ${result.sha256_after ?? result.sha256 ?? 'a new hash'}.`
				: `Observation accepted; no persistent write occurred (${result.reason ?? 'unchanged'}).`;
			await this.refresh();

			return result;
		} catch (error) {
			this.error = error instanceof Error ? error.message : String(error);

			throw error;
		} finally {
			this.loading = false;
		}
	}

	async teachSequence(
		text: string,
		mode: ApiSidecarLearningMode,
		chunkSize: number
	): Promise<ApiSidecarSequenceResponse> {
		const selected = this.selectedSidecar;

		if (!selected?.loaded) throw new Error('Select a loaded sidecar first.');
		if (!selected.learning_enabled) throw new Error('Enable persistent learning first.');
		if (!text) throw new Error('Enter an experience to teach.');
		if (mode === 'chunked' && (!Number.isInteger(chunkSize) || chunkSize < 1)) {
			throw new Error('Chunk size must be a positive integer.');
		}

		this.loading = true;
		this.error = null;
		this.message = null;
		try {
			const result = await SidecarService.observeSequence({
				id: selected.id,
				text,
				mode,
				...(mode === 'chunked' ? { chunk_size: chunkSize } : {})
			});
			const tokens = result.tokens_observed ?? result.sidecar_experience_events ?? 0;
			const writes = result.writes_realized ?? result.sidecar_experience_events ?? 0;
			const speed = result.observed_tokens_per_second;
			this.message = `Observed ${String(tokens)} tokens; ${String(writes)} writes${
				speed === undefined ? '' : ` at ${speed.toFixed(1)} tok/s`
			}. State ${shortHash(result.sha256_before)} -> ${shortHash(result.sha256_after)}.`;
			await this.refresh();

			return result;
		} catch (error) {
			this.error = error instanceof Error ? error.message : String(error);
			throw error;
		} finally {
			this.loading = false;
		}
	}

	private async action(action: () => Promise<unknown>, successMessage: string): Promise<void> {
		this.loading = true;
		this.error = null;
		this.message = null;
		try {
			await action();
			this.message = successMessage;
		} catch (error) {
			this.error = error instanceof Error ? error.message : String(error);

			throw error;
		} finally {
			this.loading = false;
		}
	}
}

export const sidecarStore = new SidecarStore();

function shortHash(value: string | undefined): string {
	return value ? value.slice(0, 10) : 'n/a';
}
