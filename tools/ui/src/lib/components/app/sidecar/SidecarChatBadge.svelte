<script lang="ts">
	import { sidecarStore } from '$lib/stores/sidecar.svelte';
	import { onMount } from 'svelte';

	let selected = $derived(sidecarStore.selectedSidecar);

	onMount(() => {
		void sidecarStore.initialize();
	});
</script>

<div class="pointer-events-auto mb-2 flex justify-center">
	<span
		class="rounded-full border border-border/60 bg-background/90 px-3 py-1 text-xs text-muted-foreground shadow-sm"
	>
		{#if selected?.loaded}
			Sidecar: <span class="font-medium text-foreground">{selected.id}</span>
			<span class="ml-1">({selected.learning_enabled ? 'teaching enabled' : 'teaching disabled'})</span>
		{:else}
			No sidecar
		{/if}
	</span>
</div>
