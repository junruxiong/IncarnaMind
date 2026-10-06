import { create } from "zustand";
import { ChatSession } from "@/lib/interfaces/interface";
import { persist, devtools, createJSONStorage } from "zustand/middleware";

type SessionActions = {
  activeSession: ChatSession | null;
  sessions: ChatSession[];
  hasMore: boolean;
  addSessions: (newSessions: ChatSession | ChatSession[]) => void;
  renameSession: (sessionId: string, newName: string) => void;
  deleteSessions: (sessionIds: string | string[]) => void;
  setActiveSession: (sessionId: string | null) => void;
  resetSession: () => void;
};

// partial persist
export const useSessionStore = create<SessionActions>()(
  devtools(
    persist(
      (set) => ({
        activeSession: null,
        sessions: [],
        hasMore: true,

        addSessions: (newSessions) =>
          set((state) => ({
            sessions: Array.isArray(newSessions)
              ? [...state.sessions, ...newSessions]
              : [...state.sessions, newSessions],
          })),

        renameSession: (sessionId, newName) =>
          set((state) => ({
            sessions: state.sessions.map((session) =>
              session.id === sessionId
                ? { ...session, session_name: newName }
                : session
            ),
          })),

        deleteSessions: (sessionIds) =>
          set((state) => ({
            sessions: state.sessions.filter((session) =>
              !Array.isArray(sessionIds)
                ? session.id !== sessionIds
                : !sessionIds.includes(session.id)
            ),
          })),

        setActiveSession: (sessionId) =>
          set((state) => ({
            activeSession:
              state.sessions.find((session) => session.id === sessionId) ||
              null,
          })),

        resetSession: () => {
          set(() => ({
            activeSession: null,
          }));
        },
      }),
      {
        name: "session-store",
        getStorage: () => sessionStorage,
        // storage: createJSONStorage(() => sessionStorage),
        partialize: (state) => ({
          activeSession: state.activeSession,
        }),
        skipHydration: true,
      }
    )
  )
);
