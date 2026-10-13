import { AsyncLocalStorage } from "node:async_hooks";

/**
 * The Task whose work is running, through every await below it: the meter
 * keys each model request to it (`model_requests.run_id`), so a Task's cost
 * can be told without passing its id through the Answer engine.
 */
export const currentTask = new AsyncLocalStorage<string>();
