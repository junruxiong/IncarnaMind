import { create } from "zustand";
import {
  persist,
  devtools,
  createJSONStorage,
  StateStorage,
} from "zustand/middleware";

import { Message, OrderItem, GroupedMessage } from "../interfaces/interface";

interface MessageStates {
  messages: Record<string, Message>; // Changed to an object
  orders: OrderItem[];
  addMessages: (newMessages: Message[]) => void;
  updateMessage: (clientId: string, newMessage: string) => void;
  orderMessages: () => void;
  deleteMessages: (clientIds: string[]) => void;
  clearStates: () => void;
  setOrder: (newOrders: OrderItem[]) => void;
}

export const useMessageStore = create<MessageStates>()(
  devtools(
    // persist(
    (set) => ({
      messages: {},
      orders: [],

      addMessages: (newMessages) =>
        set((state) => {
          const updatedMessages = { ...state.messages };
          const messagesArray = Array.isArray(newMessages)
            ? newMessages
            : [newMessages];
          messagesArray.forEach((message) => {
            if (message.client_id) {
              updatedMessages[message.client_id] = message;
            }
          });
          return { ...state, messages: updatedMessages };
        }),

      updateMessage: (clientId, newMessage) =>
        set((state) => {
          const updatedMessages = { ...state.messages };

          if (updatedMessages[clientId]) {
            updatedMessages[clientId] = {
              ...updatedMessages[clientId],
              message_text: newMessage,
            };
          }
          return { ...state, messages: updatedMessages };
        }),

      orderMessages: () =>
        set((state) => {
          const updatedOrder: OrderItem[] = [];
          Object.values(state.messages).forEach((message) => {
            let orderEntry = updatedOrder.find(
              (o) => o.order === message.order
            );
            if (orderEntry) {
              orderEntry.client_ids.push(message.client_id);
            } else {
              updatedOrder.push({
                order: message.order,
                client_ids: [message.client_id],
              });
            }
          });

          updatedOrder.forEach((orderItem) => {
            orderItem.client_ids.sort((a, b) => {
              const dateA = new Date(state.messages[a].created_at);
              const dateB = new Date(state.messages[b].created_at);
              return dateA.getTime() - dateB.getTime();
            });
          });

          updatedOrder.sort((a, b) => Number(a.order) - Number(b.order));

          return { ...state, orders: updatedOrder };
        }),

      setOrder: (newOrders) =>
        set((state) =>
          // replace order with new order
          ({ ...state, orders: newOrders })
        ),

      deleteMessages: (clientIds) =>
        set((state) => {
          const updatedMessages = { ...state.messages };
          const updatedOrder = state.orders
            .map((orderItem) => ({
              ...orderItem,
              client_ids: orderItem.client_ids.filter(
                (clientId) => !clientIds.includes(clientId)
              ),
            }))
            .filter((orderItem) => orderItem.client_ids.length > 0);

          clientIds.forEach((clientId) => {
            delete updatedMessages[clientId];
          });

          return { ...state, messages: updatedMessages, orders: updatedOrder };
        }),

      clearStates: () =>
        set({
          messages: {},
          orders: [],
        }),
    })
    // {
    //   name: "message-store",
    //   // getStorage: () => sessionStorage,
    //   storage: createJSONStorage(() => sessionStorage),
    //   // partialize: (state) => ({
    //   //   activeSession: state.activeSession,
    //   // }),
    //   skipHydration: true,
    // }
    // )
  )
);

interface ContentActions {
  content: string;
  setContent: (content: Message[]) => void;
  appendContent: (content: Message[]) => void;
  resetContent: () => void;
}

export const useContentStore = create<ContentActions>()(
  devtools(
    // persist(
    (set) => ({
      content: "",
      setContent: (content) =>
        set(() => {
          const groupedData = transformData(content);
          const htmlContent = generateHtmlContent(groupedData);
          return { content: htmlContent };
        }),
      appendContent: (content) =>
        set((state) => {
          const groupedData = transformData(content);
          const htmlContent = generateHtmlContent(groupedData);
          return { content: state.content + htmlContent };
        }),
      resetContent: () => set({ content: "" }),
    })
    // ,{
    //   name: "content-store",
    //   // getStorage: () => sessionStorage,
    //   storage: createJSONStorage(() => sessionStorage),
    //   // partialize: (state) => ({
    //   //   activeSession: state.activeSession,
    //   // }),
    //   skipHydration: true,
    // }
    // )
  )
);

const transformData = (data: Message[]): Record<string, any[]> => {
  const sortedData = data.sort((a, b) => a.order - b.order);
  const groupedData: Record<string, any[]> = {};

  sortedData.forEach((item) => {
    if (item.message_type === "text") {
      const key = `text-${item.order}`;
      if (!groupedData[key]) {
        groupedData[key] = [];
      }
      groupedData[key].push({ type: "text", data: item });
    } else {
      // Group both query and output types under the same key based on the order
      const key = `qa-${item.order}`;
      if (!groupedData[key]) {
        groupedData[key] = [];
      }
      groupedData[key].push(item);
    }
  });

  return groupedData;
};

const generateHtmlContent = (groupedData: Record<string, any[]>): string => {
  return Object.values(groupedData)
    .map((items) => {
      if (items[0].type === "text") {
        return `<text-container>${items
          .map(
            (item) =>
              `<text-component data-id='${item.data.client_id}'><p>${item.data.message_text}</p></text-component>`
          )
          .join("")}</text-container>`;
      } else {
        return `<qa-container>
          ${items
            .map((item: Message) => {
              const content = !item.message_text
                ? "<p></p>"
                : item.message_text;
              if (item.message_type === "query") {
                return `<query-component data-id='${item.client_id}'>${content}</query-component>`;
              } else if (item.message_type === "output") {
                return `<output-component data-id='${item.client_id}'>${content}</output-component>`;
              }
            })
            .join("")}
        </qa-container>`;
      }
    })
    .join("");
};
