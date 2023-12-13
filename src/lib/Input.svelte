<script lang="ts">
	import { message, predicting } from "$lib";
	import { open } from "@tauri-apps/api/dialog";
	import { listen, type Event, type UnlistenFn } from "@tauri-apps/api/event";
	import { basename } from "@tauri-apps/api/path";
	import { convertFileSrc, invoke } from "@tauri-apps/api/tauri";
	import { onDestroy, onMount } from "svelte";

	const models = ["CUSTOM", "VGG16"] as const;
	let model_name = models[0];
	let prediction_file: string;
	let prediction_file_caption: string;

	async function assign_file(selected_file: string | string[] | null): Promise<boolean> {
		if (!selected_file) {
			$message = { style: "text-error", value: "Please upload a file for Prediction" };
			return false;
		}

		if (typeof selected_file !== "string") {
			if (selected_file.length > 1) {
				$message = {
					style: "text-warning",
					value: "Only the last file uploaded will be used for prediction.",
				};
			}
			selected_file = selected_file[0];
		}

		prediction_file = selected_file;
		prediction_file_caption = await basename(prediction_file);

		return true;
	}

	async function select_file() {
		let selected_file = await open({
			multiple: false,
			filters: [{ name: "Images", extensions: ["jpg", "png", "jpeg"] }],
		});

		if (!(await assign_file(selected_file))) return;
	}

	async function load_file(file: string): Promise<string> {
		return new Promise((resolve) => resolve(convertFileSrc(file)));
	}

	let unlisten_drop: UnlistenFn;

	function pick_file(e: Event<string[]>) {
		assign_file(e.payload);
	}

	onMount(async () => {
		unlisten_drop = await listen("tauri://file-drop", pick_file);
	});

	onDestroy(() => {
		if (unlisten_drop) unlisten_drop();
	});

	async function invoke_predictor() {
		if (!prediction_file) {
			$message = { style: "text-error", value: "Please upload a file for Prediction" };
			return;
		}
		if (!model_name) {
			$message = { style: "text-error", value: "Please select a model for Prediction" };
			return;
		}
		$predicting = true;
		$message = { value: "Predicting... " };
		await invoke("predictor", { model: model_name, file: prediction_file });
	}
</script>

<div class="flex flex-col p-4 gap-2 rounded-t-2xl lg:rounded-none lg:rounded-l-2xl h-1/2 lg:w-1/2 lg:h-full">
	<div class="flex justify-between items-center w-full">
		<div class="flex items-center gap-2">
			<iconify-icon width="1.5rem" icon="line-md:image-twotone" />
			<h1 class="text-sm md:text-base">Input</h1>
		</div>
		<div class="join">
			<select
				name="model"
				bind:value={model_name}
				class="select select-bordered select-xs md:select-sm focus:outline-none join-item"
			>
				<option selected disabled>Models</option>
				{#each models as model}
					<option>{model}</option>
				{/each}
			</select>
			<button
				aria-busy={$predicting}
				disabled={$predicting}
				class="btn btn-xs md:btn-sm w-20 join-item border border-base-content border-opacity-20"
				on:click={invoke_predictor}
			>
				{#if !$predicting}
					Predict
				{:else}
					<iconify-icon icon="line-md:loading-twotone-loop" />
				{/if}
			</button>
		</div>
	</div>

	<button
		class="grid w-full h-full overflow-hidden bg-base-200 rounded-box place-items-center"
		on:click={select_file}
	>
		{#if !prediction_file}
			<div class="flex flex-col items-center">
				<iconify-icon height="2.5rem" icon="line-md:upload-loop" />
				<span>Upload Image</span>
			</div>
		{:else}
			{#await load_file(prediction_file)}
				<iconify-icon height="2.5rem" icon="line-md:image-twotone" />
			{:then url}
				<figure>
					<img class="h-44 md:h-52 lg:h-64 rounded" src={url} alt={prediction_file_caption} />
					<figcaption class="text-sm md:text-base italic">
						{prediction_file_caption}
					</figcaption>
				</figure>
			{:catch}
				<p class="text-xs md:text-base font-semibold text-error-content">Failed to upload prediction file</p>
			{/await}
		{/if}
	</button>
</div>
