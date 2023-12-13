<script lang="ts">
	import Input from "$lib/Input.svelte";
	import Output from "$lib/Output.svelte";
	// import type { ActionData } from './$types';

	// export let form: ActionData;
</script>

<div class="flex flex-col w-full h-[100dvh] items-center justify-center gap-4 bg-base-300">
	<h1 class="text-xl font-semibold">Ophthalmic Classification</h1>
	<div class="flex rounded-2xl border border-neutral w-11/12 h-5/6 lg:w-5/6 bg-base-100 shadow-xl">
		<div class="flex h-full w-full flex-col lg:flex-row">
			<Input />
			<div class="divider m-0 lg:divider-horizontal lg:m-0" />
			<!-- <Output {form} /> -->
			<Output />
		</div>
	</div>
</div>
