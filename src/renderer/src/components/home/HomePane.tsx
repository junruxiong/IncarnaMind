import { ComingUp } from "./ComingUp";
import { FolderCards } from "./FolderCards";
import { HomeHeader } from "./HomeHeader";
import { RecentMinds } from "./RecentMinds";

export function HomePane() {
  return (
    <section
      data-testid="home"
      className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden bg-sheet text-ui text-ink"
    >
      <div className="@container/home min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-[760px] px-6 pt-10 pb-24 @max-[760px]/home:px-4 @max-[760px]/home:pt-8">
          <HomeHeader />
          <ComingUp />
          <RecentMinds />
          <FolderCards />
        </div>
      </div>
    </section>
  );
}
