import { useCallback, useEffect, useState } from "react";
import { getSessions } from "../service/session";
import { useSessionStore } from "../stores/sessionStore";

export const useSessions = () => {
  const [page, setPage] = useState(1);
  const sessions = useSessionStore((state) => state.sessions);
  //   console.log("sessions+++++", sessions);

  const setPageCallback = useCallback(
    (newPage: any) => {
      setPage(newPage);
    },
    [page]
  );

  useEffect(() => {
    async function fetchSessions() {
      try {
        const { results, next } = await getSessions(page);
        // console.log("next------", next);
        sessions.length === 0 &&
          useSessionStore.setState({ sessions: results });
        page > 1 && useSessionStore.getState().addSessions(results);
        // if next is null, hasMore is false
        next ?? useSessionStore.setState({ hasMore: false });
      } catch (error) {
        console.log(error);
      }
    }
    fetchSessions();
  }, [page]);

  return { setPageCallback };
};
