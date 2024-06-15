import { useMemo } from 'react';
import { createBrowserClient } from '@supabase/ssr';

export function getSupabaseBrowserClient() {
  return createBrowserClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
  );
}

function useSupabaseClient() {
  return useMemo(getSupabaseBrowserClient, []);
}

export default useSupabaseClient;
