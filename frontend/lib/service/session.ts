import { API } from "./axios";

export const getSessions = async (page: number) => {
  const { data } = await API.get(`/chat/?page=${page}`, {
    headers: {
      Authorization: `Bearer ${localStorage.getItem("access")}`,
    },
  });
  return data;
};

export const addSession = async () => {
  const { data } = await API.post("/chat/", {
    headers: {
      Authorization: `Bearer ${localStorage.getItem("access")}`,
    },
  });
  return data;
};

export const changeSessionName = async (sessionId: string, newName: string) => {
  const { data } = await API.patch(
    `/chat/${sessionId}/`,
    {
      session_name: newName,
    },
    {
      headers: {
        Authorization: `Bearer ${localStorage.getItem("access")}`,
      },
    }
  );
  return data;
};

export const deleteSession = async (sessionId: string) => {
  const { data } = await API.delete(`/chat/${sessionId}/`, {
    headers: {
      Authorization: `Bearer ${localStorage.getItem("access")}`,
    },
  });
};
