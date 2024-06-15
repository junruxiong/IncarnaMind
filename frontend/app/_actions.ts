"use server";

import createSupabaseServerClient from "@/lib/supabase/server";
import { CreateUserInput, LoginUserInput } from "@/lib/user-schema";

export async function signUpWithEmailAndPassword({
  data,
  emailRedirectTo,
}: {
  data: CreateUserInput;
  emailRedirectTo?: string;
}) {
  const supabase = await createSupabaseServerClient();
  const result = await supabase.auth.signUp({
    email: data.email,
    password: data.password,
    options: {
      emailRedirectTo,
    },
  });
  return JSON.stringify(result);
}

export async function signInWithEmailAndPassword(data: LoginUserInput) {
  const supabase = await createSupabaseServerClient();
  const result = await supabase.auth.signInWithPassword({
    email: data.email,
    password: data.password,
  });
  return JSON.stringify(result);
}
