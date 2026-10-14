/**
 * "Choose another model" from an Answer that couldn't add Citations (#765)
 * opens the composer's model menu: each request counts up, and the composer's
 * model chip opens its menu when the count changes.
 */
import { create } from "zustand";

export const useModelMenuRequests = create<{ count: number }>()(() => ({ count: 0 }));

export const requestModelMenu = () =>
  useModelMenuRequests.setState((state) => ({ count: state.count + 1 }));
