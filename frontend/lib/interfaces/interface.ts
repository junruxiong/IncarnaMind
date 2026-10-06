export interface User {
  id: string;
  chat_sessions: ChatSession[];
  last_login: Date | null;
  is_superuser: boolean;
  username: string;
  first_name: string;
  last_name: string;
  email: string;
  is_staff: boolean;
  is_active: boolean;
  date_joined: Date;
  salt: string | null; //! remove null later
  verifier: string | null; //! remove null later
  groups: any[];
  user_permissions: any[];
}

export interface ChatSession {
  id: string;
  user: string;
  session_name: string;
  created_at: Date;
}

export interface FileMetadata {
  id: string;
  // file: string;
  original_filename: string;
  md5: string;
  created_at: Date;
  updated_at: Date;
}

export interface CacheFiles {
  id: string;
  data: any;
}

export interface Message {
  client_id: string;
  id: string | null;
  order: number;
  user: string | null | undefined;
  session_id: string | null;
  message_type: string;
  message_text: string;
  metadata: Record<string, string> | null;
  created_at: Date;
  updated_at: Date;
  in_reply_to: null | string;
}

export interface OrderItem {
  order: number;
  client_ids: string[];
}

export interface GroupedMessageItem {
  id: string;
  user: string;
  message_type: string;
  message_text: string;
  metadata: Record<string, string>;
  created_at: Date;
  updated_at: Date;
  session_id: string;
  in_reply_to: string | null;
}

export interface GroupedMessage {
  order: number;
  items: GroupedMessageItem[];
}

export interface reorderedIds {
  client_id: string | null;
  order: number;
}

// interface Metadata {
//   [key: string]: string;
// }

export interface AuthState {
  access: string | null;
  refresh: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  user: User | null;
}

export interface Form {
  [key: string]: string;
}

export interface Segment {
  lefBarWidth: number;
  viewerWidth: number;
  prevviewerWidth: number;

  rodWidth: number;
  isViewerVisible: boolean;

  minChatWidth: number;
  minLefBarWidth: number;
  minviewerWidth: number;
}
