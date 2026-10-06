"use client";
import { ChangeEvent, FormEvent, useRef, useState } from "react";
import { Form } from "@/lib/interfaces/interface";
import { login, register } from "@/lib/service/auth";
import { useAuthStore } from "@/lib/stores/authStore";
import toast, { Toaster } from "react-hot-toast";
import Link from "next/link";
import { useAuth } from "@/lib/hooks/useAuth";

export const AuthForm = ({ process }: { process: string }) => {
  const [form, setForm] = useState<Form>({});
  const { pathname, router, authChecked } = useAuth();

  const handleChange = (
    e: ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
  ) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (pathname === "/login") {
      try {
        const { message, user } = await login(form);
        // update the user state
        useAuthStore.getState().login(user);
        toast.success(message, { duration: 1000 });
        // redirect to / page
        setTimeout(() => {
          toast.dismiss();
          router.push("/");
        }, 1500);
      } catch (error: any) {
        toast.error(error.response.data.message, { duration: 2500 });
      }
    } else {
      // register
      try {
        const formData = new FormData();
        formData.append("username", form.username);
        formData.append("password", form.password);
        formData.append("email", form.email);
        formData.append("first_name", form.first_name || "");
        formData.append("last_name", form.last_name || "");

        const { data, status } = await register(formData);
        if (status === 201) {
          const user = { username: data.username, password: form.password };
          const { user: loggedUser } = await login(user);
          // update the user state
          useAuthStore.getState().login(loggedUser);
          toast.success("Successfully registered", { duration: 1000 });
          // redirect to / page
          setTimeout(() => {
            toast.dismiss();
            router.push("/");
          }, 1500);
        }
      } catch (error: any) {
        toast.error(error.response.data.message, { duration: 2500 });
      }
    }
  };

  return (
    <>
      {authChecked ? (
        <div className="h-screen w-screen flex items-center justify-center">
          <form
            className="relative flex flex-col w-[580px] justify-center gap-y-5 bg-slate-100"
            onSubmit={handleSubmit}
          >
            <h1 className="text-3xl font-bold">{process}</h1>
            <input
              type="text"
              name="username"
              placeholder="Username"
              className="input"
              required
              autoFocus
              onChange={handleChange}
            />
            <input
              type="password"
              placeholder="Password"
              name="password"
              className="input"
              required
              onChange={handleChange}
            />
            {pathname === "/register" && (
              <>
                <input
                  type="email"
                  placeholder="Email"
                  name="email"
                  className="input"
                  required
                  onChange={handleChange}
                />
                <input
                  type="text"
                  placeholder="First Name"
                  name="first_name"
                  className="input"
                  onChange={handleChange}
                />
                <input
                  type="text"
                  placeholder="Last Name"
                  name="last_name"
                  className="input"
                  onChange={handleChange}
                />
              </>
            )}

            {pathname === "/login" ? (
              <Link
                href={"/register"}
                className="text-[13px] font-semibold text-blue-600 self-end"
              >
                No account? Register here
              </Link>
            ) : (
              <Link
                href={"/login"}
                className="text-[13px] font-semibold text-blue-600 self-end"
              >
                Already have an account? Login here
              </Link>
            )}

            <button className="formButton">{process}</button>
          </form>
          <Toaster />
        </div>
      ) : (
        <></>
      )}
    </>
  );
};
