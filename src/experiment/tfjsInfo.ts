import tf from "@tensorflow/tfjs-node";
import type { TfjsInfo } from "./telemetry.ts";

/**
 * The one place the harness asks tfjs about itself. Kept apart from `telemetry.ts` so that
 * module stays importable without the native binding; import this only from code that is about
 * to run a model anyway.
 */
export function tfjsInfo(): TfjsInfo {
  return {
    version: tf.version["tfjs-node"] ?? tf.version_core,
    backend: tf.getBackend(),
  };
}
