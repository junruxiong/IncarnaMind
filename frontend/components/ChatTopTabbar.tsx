import { useSessions } from "@/lib/hooks/useSession";
import { useSessionStore } from "@/lib/stores/sessionStore";
import { useCallback, useEffect, useRef, useState } from "react";
import { ChatSession } from "@/lib/interfaces/interface";
import Link from "next/link";
import { AiOutlineClose } from "react-icons/ai";
import { FcFaq } from "react-icons/fc";
import { changeSessionName, deleteSession } from "@/lib/service/session";
import { Modal } from "@mui/material";
import { useRouter } from "next/navigation";
// const router = useRouter();

export const TopTabbar = () => {
  const router = useRouter();

  const { setPageCallback: setPage } = useSessions();
  const {
    sessions,
    activeSession,
    hasMore,
    addSessions,
    renameSession,
    deleteSessions,
    setActiveSession,
  } = useSessionStore();

  const scrollContainer = useRef<HTMLDivElement>(null);
  const observerRef = useRef<IntersectionObserver>();

  const [currentSession, setCurrentSession] = useState<ChatSession | null>(
    null
  ); // State to track the current session
  const [isEditing, setIsEditing] = useState(false);
  const [editedSession, seteditedSession] = useState("");
  const [editingSessionWidth, setEditingSessionWidth] = useState<Number | null>(
    null
  );

  // State for Modal
  const [openModal, setOpenModal] = useState(false);

  // Handle Modal
  const handleModalOpen = () => {
    setOpenModal(true);
  };

  const handleModalClose = () => {
    setOpenModal(false);
  };

  const handleDoubleClick = (
    session: ChatSession,
    event: React.MouseEvent<HTMLElement>
  ) => {
    const buttonWidth = event.currentTarget.offsetWidth + 16;
    console.log("buttonWidth", buttonWidth);

    // set the setEditingSessionWidth to the width of the button
    setEditingSessionWidth(buttonWidth);

    // Your existing code...
    seteditedSession(session.session_name);
    setIsEditing(true);
    setCurrentSession(session);
  };

  const handleSaveSessionName = async () => {
    if (
      currentSession &&
      currentSession.session_name !== editedSession &&
      editedSession !== ""
    ) {
      try {
        await changeSessionName(currentSession.id, editedSession);
        renameSession(currentSession.id, editedSession);
        setIsEditing(false);
        // update activeSession's name if the session is active
        if (activeSession && activeSession.id === currentSession.id) {
          setActiveSession(currentSession.id);
        }
      } catch (error) {
        console.error("Error updating session name:", error);
      }
    } else {
      setIsEditing(false);
    }
  };

  const handleDeleteSession = async () => {
    if (currentSession) {
      try {
        await deleteSession(currentSession.id);
        deleteSessions(currentSession.id);
        // if activeSession is the session being deleted, set activeSession to null
        if (activeSession && activeSession.id === currentSession.id) {
          setActiveSession(null);
        }
      } catch (error) {
        console.error("Error deleting session:", error);
      }
    }
  };

  const handleCreateNewMind = () => {
    // Set the active session to null, if click new session
    setActiveSession(null);
  };

  const handleWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    const container = e.currentTarget;
    const scrollAmount = e.deltaY;
    container.scrollLeft += scrollAmount;
  };

  const handleObserver = (entries: IntersectionObserverEntry[]) => {
    if (entries[0].isIntersecting && hasMore) {
      setPage((prevPage: number) => prevPage + 1);
    }
  };

  const lastTabRef = useCallback(
    (node: HTMLAnchorElement) => {
      if (observerRef.current) observerRef.current.disconnect();
      observerRef.current = new IntersectionObserver(handleObserver);
      if (node) observerRef.current.observe(node);
    },
    [hasMore] // Dependencies for the callback
  );

  return (
    <div className="flex flex-row h-10">
      <div className="group relative inline-flex items-center mt-2">
        <button
          title="Create a New Mind"
          className="whitespace-nowrap text-gray-600 text-sm font-normal h-8 pl-2 pr-3 rounded-t-[9px]"
          onClick={handleCreateNewMind}
        >
          <Link href={`/`}>
            <p className="hover:newMind">+ New Mind</p>
          </Link>
        </button>
      </div>

      <div
        ref={scrollContainer}
        onWheel={handleWheel}
        className="group relative inline-flex mt-2 mr-2 overflow-x-auto hide-scrollbar"
        style={{ flexGrow: 1, flexShrink: 1, flexBasis: "auto" }}
      >
        {sessions.map((session: ChatSession, index) => (
          <Link
            href={`/m/${session.id}`}
            key={session.id}
            onClick={() => setActiveSession(session.id)}
            ref={index === sessions.length - 1 ? lastTabRef : null}
            className={`flex items-center relative whitespace-nowrap text-sm font-normal h-full rounded-t-[9px] ${
              activeSession && activeSession.id === session.id
                ? "bg-white text-gray-700 session-active"
                : "text-gray-600 hover:bg-gray-100 session-inactive"
            } ${
              !activeSession ||
              (activeSession.id !== session.id &&
                (!sessions[index - 1] ||
                  sessions[index - 1].id !== activeSession.id))
                ? "custom-border"
                : ""
            }`}
          >
            <FcFaq className="absolute left-[10px] text-base"></FcFaq>

            {isEditing && currentSession?.id === session.id ? (
              // Show input field if the session is being edited
              <>
                <input
                  type="text"
                  value={editedSession}
                  onChange={(e) => seteditedSession(e.target.value)}
                  onBlur={handleSaveSessionName} // Save on losing focus
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      handleSaveSessionName();
                    }
                  }}
                  className="ml-8 mr-3 overflow-hidden whitespace-nowrap outline-gray-400 outline-1 text-gray-700"
                  // set editingSessionWidth as the width of the input
                  style={{ width: `${editingSessionWidth}px` }}
                  autoFocus
                />
              </>
            ) : (
              // Show session name with double click handler
              <>
                <button
                  onDoubleClick={(e) => handleDoubleClick(session, e)}
                  className="max-w-[150px] ml-8 mr-7 overflow-hidden text-ellipsis whitespace-nowrap"
                  title={session.session_name}
                >
                  {session.session_name}
                </button>
                <button
                  onClick={(e) => {
                    e.preventDefault(); // Prevent default action
                    e.stopPropagation();
                    setCurrentSession(session);
                    handleModalOpen();
                  }}
                  className="absolute right-[8px] text-sm text-gray-500 p-[2px] hover:text-gray-600 hover:bg-gray-200 rounded-[9px]"
                >
                  <AiOutlineClose />
                </button>
              </>
            )}
          </Link>
        ))}
      </div>

      <Modal open={openModal} onClose={handleModalClose}>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 max-w-4xl bg-white shadow-custom_focus p-4 rounded-[9px]">
          <h2 className="text-lg font-semibold">Delete Mind</h2>
          {currentSession && (
            <div className="mt-2">
              This action will delete{" "}
              <span className="font-medium">
                {currentSession?.session_name}
              </span>
              .
            </div>
          )}

          <div className="flex justify-end mt-4">
            <button
              onClick={handleModalClose}
              className="border border-solid border-gray-300 rounded-[9px] px-4 py-2 mr-2 hover:bg-gray-100"
            >
              Cancel
            </button>
            <button
              onClick={() => {
                // Add logic to delete the session
                if (currentSession) {
                  handleDeleteSession();
                }
                handleModalClose();
                router.push("/");
              }}
              className=" text-white bg-red-700 rounded-[9px] px-4 py-2 hover:bg-red-800"
            >
              Delete
            </button>
          </div>
        </div>
      </Modal>
    </div>
  );
};
