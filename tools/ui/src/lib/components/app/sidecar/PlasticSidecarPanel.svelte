<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { sidecarStore } from '$lib/stores/sidecar.svelte';
	import type { ApiSidecarLearningMode, ApiSidecarSequenceResponse } from '$lib/types/api';
	import { onMount } from 'svelte';

	let createId = $state('friend-test');
	let loadId = $state('friend-test');
	let loadPath = $state('');
	let savePath = $state('');
	let teachText = $state('The room costs five gold pieces.');
	let learningMode = $state<ApiSidecarLearningMode>('batch');
	let chunkSize = $state(16);
	let teachResult = $state<ApiSidecarSequenceResponse | null>(null);
	let createLearning = $state(false);

	let selected = $derived(sidecarStore.selectedSidecar);

	onMount(() => {
		void sidecarStore.initialize();
	});

	function selectSidecar(event: Event) {
		sidecarStore.select((event.currentTarget as HTMLSelectElement).value);
	}

	function setLearning(event: Event) {
		void sidecarStore
			.setLearning((event.currentTarget as HTMLInputElement).checked)
			.catch(() => {});
	}

	function teach() {
		teachResult = null;
		void sidecarStore
			.teachSequence(teachText, learningMode, chunkSize)
			.then((result) => {
				teachResult = result;
			})
			.catch(() => {});
	}

	function create() {
		void sidecarStore.create(createId.trim(), createLearning).catch(() => {});
	}

	function load() {
		void sidecarStore.load(loadId.trim(), loadPath.trim(), createLearning).catch(() => {});
	}

	function save() {
		void sidecarStore.save(savePath.trim()).catch(() => {});
	}
</script>

<section class="rounded-lg border border-border/60 bg-background/60 p-4 shadow-sm">
	<div class="mb-4 flex flex-wrap items-start justify-between gap-3">
		<div>
			<h2 class="text-base font-semibold">Plastic Sidecar</h2>
			<p class="mt-1 max-w-2xl text-sm text-muted-foreground">
				Optional external state for frozen Qwen. Chat reads the selected sidecar; persistent
				learning happens only through an explicit observation.
			</p>
		</div>
		<span class="rounded-full border border-border/60 px-2 py-1 text-xs text-muted-foreground">
			{sidecarStore.enabled ? 'Enabled' : 'Unavailable'}
		</span>
	</div>

	{#if !sidecarStore.enabled}
		<p class="rounded-md bg-muted/40 p-3 text-sm text-muted-foreground">
			Start the server with <code>--sidecar-dir &lt;directory&gt;</code> to enable sidecars.
		</p>
	{:else}
		<div class="grid gap-4 md:grid-cols-2">
			<label class="space-y-1 text-sm">
				<span class="font-medium">Active sidecar</span>
				<select
					class="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
					value={sidecarStore.selectedId}
					onchange={selectSidecar}
				>
					<option value="">No sidecar</option>
					{#each sidecarStore.sidecars as sidecar (sidecar.id)}
						<option value={sidecar.id}>
							{sidecar.id}{sidecar.loaded ? '' : ' (unloaded)'}
						</option>
					{/each}
				</select>
			</label>

			{#if selected}
				<div class="rounded-md border border-border/50 p-3 text-xs text-muted-foreground">
					<div class="font-medium text-foreground">{selected.id}</div>
					<div>
						{selected.loaded ? 'Loaded' : 'Unloaded'} · {selected.components} components · {selected.writes ??
							0} writes
					</div>
					<div class="truncate">SHA256: {selected.sha256 || 'not available'}</div>
					<div>Profile: {selected.injection_profile_id ?? 'server default'}</div>
				</div>
			{/if}
		</div>

		<div class="mt-4 grid gap-4 border-t border-border/40 pt-4 md:grid-cols-2">
			<div class="space-y-2">
				<h3 class="text-sm font-medium">Create</h3>
				<Input bind:value={createId} placeholder="friend-test" aria-label="New sidecar ID" />
				<label class="flex items-center gap-2 text-xs text-muted-foreground">
					<input type="checkbox" bind:checked={createLearning} />
					Enable persistent learning
				</label>
				<Button size="sm" onclick={create} disabled={sidecarStore.loading || !createId.trim()}
					>Create</Button
				>
			</div>

			<div class="space-y-2">
				<h3 class="text-sm font-medium">Load</h3>
				<div class="grid gap-2 sm:grid-cols-2">
					<Input bind:value={loadId} placeholder="sidecar ID" aria-label="Loaded sidecar ID" />
					<Input
						bind:value={loadPath}
						placeholder="package path"
						aria-label="Sidecar package path"
					/>
				</div>
				<p class="text-xs text-muted-foreground">
					Path is resolved by the server's sidecar directory.
				</p>
				<Button
					size="sm"
					variant="outline"
					onclick={load}
					disabled={sidecarStore.loading || !loadId.trim() || !loadPath.trim()}>Load</Button
				>
			</div>
		</div>

		<div class="mt-4 flex flex-wrap gap-2 border-t border-border/40 pt-4">
			<Button
				size="sm"
				variant="outline"
				onclick={save}
				disabled={sidecarStore.loading || !selected?.loaded}>Save</Button
			>
			<Input
				class="max-w-xs"
				bind:value={savePath}
				placeholder="package path (defaults to ID)"
				aria-label="Save package path"
			/>
			<Button
				size="sm"
				variant="outline"
				onclick={() => void sidecarStore.reset().catch(() => {})}
				disabled={sidecarStore.loading || !selected?.loaded}>Reset</Button
			>
			<Button
				size="sm"
				variant="outline"
				onclick={() => void sidecarStore.unload().catch(() => {})}
				disabled={sidecarStore.loading || !selected?.loaded}>Unload</Button
			>
		</div>

		{#if selected?.loaded}
			<div class="mt-4 rounded-md border border-border/40 p-3">
				<label class="flex items-center gap-2 text-sm">
					<input type="checkbox" checked={selected.learning_enabled} onchange={setLearning} />
					Explicit teaching enabled
				</label>
				<p class="mt-1 text-xs text-muted-foreground">
					Normal chat only reads this sidecar. Teach / Observe writes persistently when enabled.
				</p>
			</div>

			<div class="mt-4 rounded-md border border-border/40 p-3">
				<h3 class="text-sm font-medium">Teach / Observe</h3>
				<p class="mt-1 text-xs text-muted-foreground">
					Teach a complete experience in one operation. The model tokenizes it internally and keeps
					token-local targets, baselines, keys, and writes.
				</p>
				<textarea
					class="mt-3 min-h-24 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
					bind:value={teachText}
					placeholder="The room costs five gold pieces."
					aria-label="Experience text"
				></textarea>
				<div class="mt-3 grid gap-2 sm:grid-cols-2">
					<label class="space-y-1 text-xs text-muted-foreground">
						<span>Learning mode</span>
						<select
							class="h-9 w-full rounded-md border border-input bg-background px-3 text-sm text-foreground"
							bind:value={learningMode}
						>
							<option value="batch">Batch (fast)</option>
							<option value="chunked">Chunked</option>
							<option value="exact-token">Exact token (research)</option>
						</select>
					</label>
					{#if learningMode === 'chunked'}
						<label class="space-y-1 text-xs text-muted-foreground">
							<span>Chunk size</span>
							<Input type="number" min="1" step="1" bind:value={chunkSize} />
						</label>
					{/if}
				</div>
				<p class="mt-2 text-xs text-muted-foreground">
					Batch and Chunked are fast acquisition approximations. Both preserve causal token keys and
					ordered writes, but learned state is committed after the acquisition pass. Exact preserves
					token-recurrent reference behavior and is much slower.
				</p>
				<Button
					class="mt-3"
					size="sm"
					onclick={teach}
					disabled={sidecarStore.loading || !selected.learning_enabled || !teachText}
					>Learn Experience</Button
				>
				{#if teachResult}
					<div class="mt-3 grid gap-1 rounded-md bg-muted/40 p-3 text-xs text-muted-foreground sm:grid-cols-2">
						<span>Mode: {teachResult.learning_mode ?? learningMode}</span>
						<span>Tokens: {teachResult.tokens_observed ?? 'see exact response'}</span>
						<span>Writes: {teachResult.writes_realized ?? teachResult.sidecar_experience_events ?? 'n/a'}</span>
						<span>Chunks: {teachResult.chunks_processed ?? 'n/a'}</span>
						<span>Elapsed: {(teachResult.elapsed_ms ?? teachResult.total_ms)?.toFixed(1) ?? 'n/a'} ms</span>
						<span>Speed: {teachResult.observed_tokens_per_second?.toFixed(1) ?? 'n/a'} tok/s</span>
						<span class="truncate">Before: {teachResult.sha256_before ?? 'n/a'}</span>
						<span class="truncate">After: {teachResult.sha256_after ?? 'refresh status'}</span>
					</div>
				{/if}
			</div>
		{/if}

		<div class="mt-4 rounded-md border border-amber-500/30 bg-amber-500/5 p-3">
			<label class="flex items-center gap-2 text-sm font-medium">
				<input
					type="checkbox"
					checked={sidecarStore.experimentalInternalPlasticity}
					onchange={(event) =>
						sidecarStore.setExperimentalInternalPlasticity(
							(event.currentTarget as HTMLInputElement).checked
						)}
				/>
				Experimental reasoning-time plasticity
			</label>
			<p class="mt-1 text-xs text-muted-foreground">
				Research only. A request-local working copy may mutate during reasoning and is discarded
				after the response. Persistent sidecar state is not changed by this option.
			</p>
		</div>
	{/if}

	{#if sidecarStore.error}
		<p class="mt-3 text-sm text-destructive">{sidecarStore.error}</p>
	{/if}
	{#if sidecarStore.message}
		<p class="mt-3 text-sm text-muted-foreground">{sidecarStore.message}</p>
	{/if}
</section>
