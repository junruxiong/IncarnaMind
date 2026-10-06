import { API } from "./axios";
import { useMutation, useQuery } from "react-query";
import { reorderedIds } from "../interfaces/interface";

export const fetchMessages = async ({
  pageParam,
  uid, // Add uid here
}: {
  pageParam: string | number;
  uid: string | null; // Define the uid type
}) => {
  if (!uid) {
    // Handle the case when uid is null or undefined
    throw new Error("UID is required to fetch messages");
  }

  const { data } = await API.get(`/chat/${uid}/messages/?page=${pageParam}`, {
    headers: {
      Authorization: `Bearer ${localStorage.getItem("access")}`,
    },
  });
  return data;
};

export const reorderMessages = async ({
  sessionId,
  reorderedIds,
}: {
  sessionId: string | null;
  reorderedIds: reorderedIds[];
}) => {
  try {
    const { data } = await API.post(
      `/chat/${sessionId}/messages/reorder/`,
      reorderedIds,
      {
        headers: {
          Authorization: `Bearer ${localStorage.getItem("access")}`,
        },
      }
    );
    return data;
  } catch (error) {
    console.log(error);
  }
};

export const insertMessage = async ({
  sessionId,
  clientId,
  messageText,
  order,
  metadata,
  messageType,
  reorder,
}: {
  sessionId: string;
  clientId: string;
  messageText: string;
  order: number;
  metadata: Record<string, string> | null;
  messageType: string;
  reorder: reorderedIds[];
}) => {
  try {
    const { data } = await API.post(
      `/chat/${sessionId}/messages/insert_message/`,
      {
        client_id: clientId,
        message_text: messageText,
        order: order,
        metadata: metadata,
        message_type: messageType,
        reorder: reorder,
      },
      {
        headers: {
          Authorization: `Bearer ${localStorage.getItem("access")}`,
        },
      }
    );
    return data;
  } catch (error) {
    console.error(error);
    throw error;
  }
};

export const removeMessages = async ({
  sessionId,
  clientIds,
  reorder,
}: {
  sessionId: string;
  clientIds: string[];
  reorder: reorderedIds[];
}) => {
  try {
    const { data } = await API.post(
      `/chat/${sessionId}/messages/bulk_delete/`,
      {
        client_ids: clientIds,
        reorder: reorder,
      },
      {
        headers: {
          Authorization: `Bearer ${localStorage.getItem("access")}`,
        },
      }
    );
    return data;
  } catch (error) {
    console.error(error);
    throw error;
  }
};

// export const getMessages = (sessionId: string | null, page: number) => {
//   return useQuery(["messages", sessionId, page], async () => {
//     const { data } = await API.get(
//       `/chat/${sessionId}/messages/?page=${page}`,
//       {
//         headers: {
//           Authorization: `Bearer ${localStorage.getItem("access")}`,
//         },
//       }
//     );
//     return data;
//   });
// };

// export const reorderMessage = () => {
//   return useMutation(
//     async ({
//       session_id,
//       reorderedIds,
//     }: {
//       session_id: string;
//       reorderedIds: reorderedIds[];
//     }) => {
//       try {
//         const { data } = await API.post(
//           `/chat/${session_id}/messages/reorder/`,
//           reorderedIds,
//           {
//             headers: {
//               Authorization: `Bearer ${localStorage.getItem("access")}`,
//             },
//           }
//         );
//         return data;
//       } catch (error) {
//         console.error(error);
//         throw error;
//       }
//     }
//   );
// };

// export const insertMessage = () => {
//   return useMutation(
//     async ({
//       sessionId,
//       clientId,
//       messageText,
//       order,
//       metadata,
//       messageType,
//       reorder,
//     }: {
//       sessionId: string;
//       clientId: string;
//       messageText: string;
//       order: number;
//       metadata: Record<string, string> | null;
//       messageType: string;
//       reorder: reorderedIds[];
//     }) => {
//       try {
//         const { data } = await API.post(
//           `/chat/${sessionId}/messages/insert_message/`,
//           {
//             client_id: clientId,
//             message_text: messageText,
//             order: order,
//             metadata: metadata,
//             message_type: messageType,
//             reorder: reorder,
//           },
//           {
//             headers: {
//               Authorization: `Bearer ${localStorage.getItem("access")}`,
//             },
//           }
//         );
//         return data;
//       } catch (error) {
//         console.error(error);
//         throw error;
//       }
//     }
//   );
// };

// export function removeMessages() {
//   return useMutation(
//     async ({
//       sessionId,
//       clientIds,
//     }: {
//       sessionId: string;
//       clientIds: string[];
//     }) => {
//       try {
//         const { data } = await API.post(
//           `/chat/${sessionId}/messages/bulk_delete/`,
//           {
//             client_ids: clientIds,
//           },
//           {
//             headers: {
//               Authorization: `Bearer ${localStorage.getItem("access")}`,
//             },
//           }
//         );
//         return data;
//       } catch (error) {
//         console.error(error);
//         throw error;
//       }
//     }
//   );
// }

////////////////////////////////////////////////////////

// export const reorderMessages = async (
//   session_id: string,
//   reorderedIds: reorderedIds[]
// ) => {
//   try {
//     const { data } = await API.post(
//       `/chat/${session_id}/messages/reorder/`,
//       reorderedIds, // Send the array directly
//       {
//         headers: {
//           Authorization: `Bearer ${localStorage.getItem("access")}`,
//         },
//       }
//     );
//     return data;
//   } catch (error) {
//     console.log(error);
//   }
// };

// export const sendMessage = async (
//   sessionId: string,
//   sendMessage: SendMessage
// ) => {
//   const { data } = await API.post(
//     `/chat/${sessionId}/messages/`,
//     {
//       client_id: sendMessage.client_id,
//       message_text: sendMessage.message_text,
//       metadata: sendMessage.metadata,
//       metadata_crypted: sendMessage.metadata_crypted,
//     },
//     {
//       headers: {
//         Authorization: `Bearer ${localStorage.getItem("access")}`,
//       },
//     }
//   );
//   return data;
// };

// export const editMessage = async (
//   sessionId: string,
//   messageId: string,
//   message: string
// ) => {
//   const { data } = await API.patch(
//     `/chat/${sessionId}/messages/${messageId}/`,
//     {
//       message,
//     }
//   );
//   return data;
// };

// export const deleteMessage = async (sessionId: string, messageId: string) => {
//   const { data } = await API.delete(
//     `/chat/${sessionId}/messages/${messageId}/`
//   );
//   return data;
// };
