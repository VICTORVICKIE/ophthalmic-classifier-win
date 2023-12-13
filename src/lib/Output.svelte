<script lang="ts">
	import { message, predicting, type Response } from "$lib";
	import { listen, type Event as TauriEvent, type UnlistenFn } from "@tauri-apps/api/event";
	import Chart from "chart.js/auto";
	import { onDestroy, onMount } from "svelte";
	import { scale } from "svelte/transition";

	let response: Response;
	let canvas: HTMLCanvasElement;
	let canvas_wrapper: HTMLDivElement;
	let chart: Chart;
	let unlisten_prediction: UnlistenFn;

	let classes = ["ARMD", "CNV", "CSR", "DME", "DR", "DRUSEN", "GC", "MH", "NORMAL"];

	async function init_chart() {
		let probabilities: number[] = new Array(classes.length).fill(0);
		chart = new Chart(canvas, {
			type: "bar",
			data: {
				labels: classes,
				datasets: [
					{
						label: "% of Probability",
						data: probabilities,
						borderWidth: 1,
					},
				],
			},
			options: {
				responsive: true,
				maintainAspectRatio: false,
				scales: {
					y: {
						min: 0,
						max: 100,
					},
					x: {
						ticks: {
							autoSkip: false,
						},
					},
				},
				plugins: {
					legend: {
						position: "bottom",
						align: "center",
					},
					title: {
						display: true,
						text: "Probability Bar Chart",
					},
					tooltip: {
						callbacks: {
							beforeBody(tooltipItems) {
								tooltipItems[0].dataset.label = " % of Probability";
							},
						},
					},
				},
			},
		}) as Chart;
	}

	function update_chart() {
		chart.data.datasets[0].label = `${response?.result?.probability.toFixed(2)} % of Probability`;
		chart.data.datasets[0].data = response?.result?.probabilities ?? new Array(classes.length).fill(0);
		chart.update();
	}

	async function init() {
		init_chart();
		unlisten_prediction = await listen("prediction", (e: TauriEvent<string>) => {
			response = JSON.parse(e.payload);
			$message = {
				value: response.message,
				style: response.success ? "text-success" : "text-error",
			};
			$predicting = false;
		});
	}

	onMount(() => init());
	onDestroy(() => {
		if (unlisten_prediction) unlisten_prediction();
	});

	function handle_resize() {
		if (!canvas_wrapper.parentNode?.parentElement) return;
		canvas_wrapper.style.height = `${canvas_wrapper.parentNode?.parentElement?.offsetHeight - 64 - 8}px`;
		canvas_wrapper.style.width = `${canvas_wrapper.parentNode?.parentElement?.offsetWidth - 32}px`;
		canvas.height = canvas_wrapper.parentNode?.parentElement?.offsetHeight - 64 - 8;
		canvas.width = canvas_wrapper.parentNode?.parentElement?.offsetWidth - 32;
	}

	$: if (response?.success) {
		update_chart();
	}
</script>

<svelte:window on:resize={handle_resize} />
<div class="grid p-4 gap-2 rounded-b-2xl lg:rounded-none lg:rounded-r-2xl h-1/2 lg:w-1/2 lg:h-full">
	<div class="flex w-full items-center justify-between">
		<div class="flex items-center gap-2">
			<iconify-icon width="1.5rem" icon="ph:chart-bar-horizontal-duotone" /><span class="text-sm md:text-base"
				>Output</span
			>
		</div>
		<div class="flex items-center">
			{#key $message.value}
				<p in:scale class="text-xs md:text-base font-semibold {$message.style}">
					{$message.value || ""}
				</p>
			{/key}
		</div>
	</div>
	<div class="bg-base-200 rounded-box place-items-center">
		<div bind:this={canvas_wrapper} class="relative h-full w-full">
			<canvas bind:this={canvas} />
		</div>
	</div>
</div>
