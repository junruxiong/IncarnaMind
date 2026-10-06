import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { refreshToken } from "@/lib/service/auth";
import { useAuthStore } from "@/lib/stores/authStore";

export const useAuth = () => {
  const [authChecked, setAuthChecked] = useState(false);
  const router = useRouter();
  const pathname = usePathname();
  const isAuthenticated = useAuthStore((state) => state.isAuthenticated);

  useEffect(() => {
    async function verifyToken() {
      if (!isAuthenticated && localStorage.getItem("access") !== null) {
        try {
          await refreshToken();
          router.push("/");
        } catch (error) {
          localStorage.removeItem("access");
          localStorage.removeItem("refresh");
          console.log("Your session has expired");
          console.log(error);
        }
      } else {
        setAuthChecked(true);
      }
    }
    verifyToken();
  }, [isAuthenticated]);

  return { router, pathname, authChecked };
};
