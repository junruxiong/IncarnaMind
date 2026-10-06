import { API } from "./axios";

export const getFile = async (fileId: string) => {
  const { data } = await API.get(`/file/${fileId}/open_file/`, {
    responseType: "blob",
    headers: {
      Authorization: `Bearer ${localStorage.getItem("access")}`,
    },
  });
  return data;
};

export const getFileMetadata = async (page: number) => {
  const { data } = await API.get(`/file/?page=${page}`, {
    headers: {
      Authorization: `Bearer ${localStorage.getItem("access")}`,
    },
  });
  return data;
};

export const changeFileName = (fileId: string, newFilename: FormData) => {
  return API.patch(`/file/${fileId}/`, newFilename, {
    headers: {
      Authorization: `Bearer ${localStorage.getItem("access")}`,
    },
  });
};

export const deleteFile = async (fileId: string) => {
  return API.delete(`/file/${fileId}/`, {
    headers: {
      Authorization: `Bearer ${localStorage.getItem("access")}`,
    },
  });
};

export const uploadFiles = async (files: FormData) => {
  const { data } = await API.post("/file/", files, {
    headers: {
      Authorization: `Bearer ${localStorage.getItem("access")}`,
    },
  });
  return data;
};
