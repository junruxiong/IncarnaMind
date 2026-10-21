/**
 * The core's `SpeakerSeparator` (#138): each Meeting's voices are told apart in an Electron
 * utility process of its own (./speakerProcess), started for it and gone once it answers, so
 * the pass's memory (about 230 MB an hour of audio) is freed and its CPU time never holds up
 * live transcription. A crash, out of memory included, rejects, and the labels stay.
 */
import { utilityProcess } from "electron";
import type { SpeakerSeparator } from "../core";
import type { SeparationReply } from "./speakerProcess";
import speakerProcessPath from "./speakerProcess?modulePath";

export function createUtilityProcessSeparator(): SpeakerSeparator {
  return {
    separate: (request) =>
      new Promise((resolve, reject) => {
        const child = utilityProcess.fork(speakerProcessPath, [], {
          serviceName: "IncarnaMind speakers",
          stdio: "inherit",
        });
        let answered = false;
        child.on("message", (reply: SeparationReply) => {
          answered = true;
          child.kill();
          if ("turns" in reply) resolve(reply.turns);
          else reject(new Error(reply.message));
        });
        child.on("exit", (code) => {
          if (!answered) reject(new Error(`The speaker process stopped (exit code ${code}).`));
        });
        child.postMessage(request);
      }),
  };
}
