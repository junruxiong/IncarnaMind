import { RefObject, useCallback, useEffect, useRef, useState } from "react";
import { fetchMessages } from "../service/message";
import { useContentStore, useMessageStore } from "../stores/msgStore";
import { API } from "../service/axios";
import { useInfiniteQuery } from "@tanstack/react-query";

// export const useMessage = (uid: string | null) => {
//   const { messages, orders, addMessages, orderMessages, clearStates } =
//     useMessageStore();

//   const {
//     data,
//     error,
//     fetchNextPage,
//     hasNextPage,
//     isFetching,
//     isFetchingNextPage,
//     status,
//   } = useInfiniteQuery({
//     queryKey: ["messages", uid],
//     queryFn: ({ pageParam }) => fetchMessages({ pageParam, uid }),
//     initialPageParam: 1,
//     getNextPageParam: (lastPage) => {
//       return lastPage.next ? lastPage.next.split("=")[1] : undefined;
//     },
//   });

//   useEffect(() => {
//     if (!uid) {
//       // Reset messages if uid is undefined or null
//       clearStates();
//     } else if (data) {
//       // Add new messages to the store when data changes
//       const newMessages = data.pages.flatMap((page) => page.results);
//       clearStates();
//       addMessages(newMessages);
//       orderMessages();
//     }
//   }, [data, uid]);

//   return { messages, orders, fetchNextPage, hasNextPage, isFetchingNextPage };
// };

export const useContent = (uid: string | null) => {
  const { content, setContent, appendContent, resetContent } =
    useContentStore();

  const {
    data,
    error,
    fetchNextPage,
    hasNextPage,
    isFetching,
    isFetchingNextPage,
    status,
  } = useInfiniteQuery({
    queryKey: ["messages", uid],
    queryFn: ({ pageParam }) => fetchMessages({ pageParam, uid }),
    initialPageParam: 1,
    getNextPageParam: (lastPage) => {
      return lastPage.next ? lastPage.next.split("=")[1] : undefined;
    },
  });

  useEffect(() => {
    if (!uid) {
      // Reset messages if uid is undefined or null
      resetContent();
    } else if (data) {
      // Add new messages to the store when data changes
      const newMessages = data.pages.flatMap((page) => page.results);
      resetContent();
      setContent(newMessages);
    }
  }, [data, uid]);

  return { content, fetchNextPage, hasNextPage, isFetchingNextPage };
};

export const useInfiniteScroll = (
  ref: RefObject<HTMLDivElement>,
  hasNextPage: boolean | undefined,
  isFetchingNextPage: boolean,
  fetchNextPage: () => void
) => {
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && hasNextPage && !isFetchingNextPage) {
          fetchNextPage();
        }
      },
      { threshold: 1.0 }
    );

    if (ref.current) {
      observer.observe(ref.current);
    }

    return () => {
      if (ref.current) {
        observer.unobserve(ref.current);
      }
    };
  }, [hasNextPage, isFetchingNextPage, fetchNextPage, ref]);
};
