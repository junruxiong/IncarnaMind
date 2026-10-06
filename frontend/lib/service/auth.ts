import { Form } from "../interfaces/interface";
import { API } from "./axios";

export const login = async (credentials: Form) => {
  const { data } = await API.post("login/", credentials);
  localStorage.setItem("access", data.access);
  localStorage.setItem("refresh", data.refresh);
  return data;
};

export const register = async (credentials: FormData) => {
  const { data, status } = await API.post("users/", credentials);
  return { data, status };
};

export const verify = async () => {
  const { data } = await API.post("verify/", {
    token: localStorage.getItem("access"),
  });
  return data;
};

export const refreshToken = async () => {
  const { data } = await API.post("refresh/", {
    refresh: localStorage.getItem("refresh"),
  });
  localStorage.setItem("access", data.access);
  localStorage.setItem("refresh", data.refresh);
  return data;
};

export const userLogged = async () => {
  const { data } = await API.get("user-logged/", {
    headers: {
      Authorization: `Bearer ${localStorage.getItem("access")}`,
    },
  });
  return data;
};
