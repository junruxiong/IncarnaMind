import { create } from "zustand";
import { AuthState, User } from "@/lib/interfaces/interface";
import { persist, devtools } from "zustand/middleware";

type AuthActions = {
  login: (userData: User) => void;
  logout: () => void;
};

// ANCHOR these states can be concise
export const useAuthStore = create<AuthState & AuthActions>()(
  devtools(
    persist(
      (set) => ({
        access: null,
        refresh: null,
        isAuthenticated: false,
        isLoading: true,
        user: null,

        login: (userData: User) =>
          set((state) => ({
            ...state,
            isAuthenticated: true,
            isLoading: false,
            user: userData,
          })),

        logout: () =>
          set((state) => ({
            ...state,
            access: null,
            refresh: null,
            isAuthenticated: false,
            isLoading: false,
            user: null,
          })),
      }),
      {
        name: "auth-store",
        partialize: (state) => ({
          isAuthenticated: state.isAuthenticated,
          isLoading: state.isLoading,
          user: state.user,
          // remove password in user
          // user: state.user && {
          //   id: state.user.id,
          //   username: state.user.username,
          // },
        }),
        skipHydration: true,
      }
    )
  )
);
