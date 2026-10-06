import { create } from "zustand";
import { Segment } from "@/lib/interfaces/interface";
import { persist, devtools, createJSONStorage } from "zustand/middleware";

// export const useSegmentStore = create<Segment>()(
//   devtools(
//     persist(
//       (set) => ({
//         lefBarWidth: 270,
//         viewerWidth: 550,
//         prevviewerWidth: 550,
//         rodWidth: 3,

//         isViewerVisible: false,
//         minLefBarWidth: 150,
//         minviewerWidth: 220,
//         minChatWidth: 300,

//         setSegment: (segment: Segment) => set(segment),
//       }),
//       {
//         name: "segment-store",
//         storage: createJSONStorage(() => localStorage),
//         skipHydration: true,
//       }
//     )
//   )
// );

export const useSegmentStore = create<Segment>()(
  devtools(
    persist(
      (set) => ({
        lefBarWidth: 270,
        viewerWidth: 0,
        prevviewerWidth: 550,
        rodWidth: 3,

        isViewerVisible: false,
        minLefBarWidth: 165,
        minviewerWidth: 220,
        minChatWidth: 300,

        setSegment: (segment: Segment) => set(segment),
      }),
      {
        name: "segment-store",
        getStorage: () => sessionStorage,
        skipHydration: true,
      }
    )
  )
);
