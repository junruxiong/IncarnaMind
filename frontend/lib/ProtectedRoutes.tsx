"use client";
import { useEffect, useState } from "react";
import { refreshToken, userLogged, verify } from "@/lib/service/auth";
import Loading from "@/app/loading";
import { redirect } from "next/navigation";
import { useAuthStore } from "./stores/authStore";

export const ProtectedRoutes = ({
  children,
}: {
  children: React.ReactNode;
}) => {
  const [showLoading, setShowLoading] = useState(true);
  const { isAuthenticated, isLoading } = useAuthStore((state) => ({
    isAuthenticated: state.isAuthenticated,
    isLoading: state.isLoading,
  }));

  useEffect(() => {
    async function checkAuth() {
      if (!isAuthenticated && localStorage.getItem("access") !== null) {
        try {
          await refreshToken();
          await verify();

          const user = await userLogged();
          // console.log("user", user);
          useAuthStore.setState({ user, isAuthenticated: true });
        } catch (error: any) {
          localStorage.removeItem("access");
          localStorage.removeItem("refresh");

          console.log(error.response.data);
        }
      } else {
        setShowLoading(false);
      }
    }
    checkAuth();
  }, [isAuthenticated, isLoading]);
  if (showLoading) return <Loading />;
  if (!isAuthenticated && isLoading) return redirect("/login");

  return <>{children}</>;
};
