import { expect, test } from "@playwright/test";
import {
  createDataFolder,
  dismissChatSetup,
  dragBy,
  launchApp,
  openViewer,
  removeDataFolder,
  widthOf,
} from "./app";

let dataDir: string;
test.beforeEach(async () => {
  dataDir = await createDataFolder();
});
test.afterEach(async () => {
  await removeDataFolder(dataDir);
});

test("a Mind created before quitting is still there after reopening the app", async () => {
  // First run: create a Mind. It appears in the sidebar and opens in the centre.
  const first = await launchApp(dataDir);
  await dismissChatSetup(first.window);
  await first.window.getByTestId("new-mind").click();

  const created = first.window.getByTestId("mind-list-item");
  await expect(created).toHaveCount(1);
  const mindId = await created.getAttribute("data-mind-id");
  expect(mindId).toBeTruthy();
  await expect(first.window.getByTestId("mind-pane")).toHaveAttribute("data-mind-id", `${mindId}`);
  await expect(first.window.getByTestId("mind-title")).toBeVisible();
  await first.app.close();

  // Second run on the same data folder: the Mind is listed and opens.
  const second = await launchApp(dataDir);
  const restored = second.window.getByTestId("mind-list-item");
  await expect(restored).toHaveCount(1);
  await expect(restored).toHaveAttribute("data-mind-id", `${mindId}`);
  await restored.click();
  await expect(second.window.getByTestId("mind-pane")).toHaveAttribute("data-mind-id", `${mindId}`);
  await expect(second.window.getByTestId("mind-title")).toBeVisible();
  await second.app.close();
});

test("the Document viewer is hidden until opened, resizes from its left edge and keeps its width", async () => {
  const first = await launchApp(dataDir);
  const { window } = first;
  await dismissChatSetup(window);
  const viewer = window.getByTestId("viewer");
  const mindArea = window.getByTestId("mind-area");

  // Closed on launch: no panel at all, and the Mind area takes the rest of the width.
  await expect(viewer).toHaveCount(0);
  const fullWidth = await widthOf(mindArea);

  // Opening it narrows the Mind area.
  await openViewer(window);
  await expect(viewer).toBeVisible();
  expect(await widthOf(mindArea)).toBeLessThan(fullWidth);

  // Dragging the left edge 100px to the left widens the panel by 100px.
  const openedWidth = await widthOf(viewer);
  await dragBy(window, window.getByTestId("viewer-resize"), -100);
  await expect.poll(() => widthOf(viewer)).toBe(openedWidth + 100);

  // Esc closes it and the Mind area gets its full width back.
  await window.keyboard.press("Escape");
  await expect(viewer).toHaveCount(0);
  expect(await widthOf(mindArea)).toBe(fullWidth);
  await first.app.close();

  // The width is a per-device setting, so it survives a restart. The close button closes it.
  const second = await launchApp(dataDir);
  await expect(second.window.getByTestId("viewer")).toHaveCount(0);
  await openViewer(second.window);
  await expect.poll(() => widthOf(second.window.getByTestId("viewer"))).toBe(openedWidth + 100);
  await second.window.getByTestId("viewer-close").click();
  await expect(second.window.getByTestId("viewer")).toHaveCount(0);
  await second.app.close();
});

test("first-run chat setup appears on a fresh data folder and can be set up later", async () => {
  const first = await launchApp(dataDir);
  const { window } = first;

  // No provider is preselected, and no key is needed to get past this screen.
  const setup = window.getByTestId("chat-setup");
  await expect(setup).toBeVisible();
  const providerChoices = setup.getByTestId("provider-form").getByRole("radio");
  await expect(providerChoices).toHaveCount(4);
  await expect(setup.getByRole("radio", { checked: true })).toHaveCount(0);

  await setup.getByTestId("chat-setup-later").click();
  await expect(setup).toBeHidden();

  // Notes and Documents work without a provider; Questions explain what to configure.
  await window.getByTestId("new-mind").click();
  await expect(window.getByTestId("mind-pane")).toBeVisible();
  await expect(window.getByTestId("chat-readiness")).toBeVisible();
  await first.app.close();

  // "Set up later" is remembered.
  const second = await launchApp(dataDir);
  await second.window.getByTestId("mind-list-item").click();
  await expect(second.window.getByTestId("chat-readiness")).toBeVisible();
  await expect(second.window.getByTestId("chat-setup")).toBeHidden();

  // The notice opens Settings, where a provider can be set up.
  await second.window.getByTestId("chat-readiness").getByRole("button").click();
  await expect(second.window.getByTestId("chat-model-settings")).toBeVisible();
  await expect(second.window.getByTestId("consent-settings")).toBeVisible();
  await second.app.close();
});
