"use client";
import { LeftSideBar } from "@/components/Leftbar";
import { TopTabbar } from "@/components/ChatTopTabbar";
import { Viewer } from "@/components/Viewer";
import { ProtectedRoutes } from "@/lib/ProtectedRoutes";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const queryClient = new QueryClient();

function Layout({ children }: { children: React.ReactNode }) {
  return (
    <QueryClientProvider client={queryClient}>
      <ProtectedRoutes>
        <div className="flex h-screen">
          <LeftSideBar></LeftSideBar>
          <Viewer></Viewer>
          <div className="flex flex-col flex-auto bg-gray-200 overflow-hidden">
            <TopTabbar />
            {children}
          </div>
        </div>
      </ProtectedRoutes>
    </QueryClientProvider>
  );
}

export default Layout;
