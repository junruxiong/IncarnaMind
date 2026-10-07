/** Progress lines on the terminal, each with the time since the run started. */
export type Log = (message: string) => void;

export function createLog(started = Date.now()): Log {
  return (message) => {
    const seconds = ((Date.now() - started) / 1000).toFixed(0).padStart(4);
    console.log(`[eval ${seconds}s] ${message}`);
  };
}
