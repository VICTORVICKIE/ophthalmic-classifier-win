import { writable } from "svelte/store";

interface Message {
	style?: string;
	value: string;
}

export interface Prediction {
	model: string;
	prediction: string;
	probability: number;
	classes: string[];
	probabilities: number[];
}

export interface Response {
	success: boolean;
	result: Prediction | null;
	message: string;
}

export const message = writable<Message>({ style: "", value: "" });
export const predicting = writable<boolean>(false);
