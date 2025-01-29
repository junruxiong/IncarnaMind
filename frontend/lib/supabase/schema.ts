import { relations, sql } from "drizzle-orm";
import {
  boolean,
  integer,
  jsonb,
  pgTable,
  text,
  timestamp,
  uuid,
} from "drizzle-orm/pg-core";
import {
  prices,
  products,
  subscriptionStatus,
  users,
} from "../../migrations/schema";

export const workspaces = pgTable("workspaces", {
  id: uuid("id").defaultRandom().primaryKey().notNull(),
  createdAt: timestamp("created_at", {
    withTimezone: true,
    mode: "string",
  })
    .defaultNow()
    .notNull(),
  workspaceOwner: uuid("workspace_owner").notNull(),
  title: text("title").notNull(),
  iconId: text("icon_id").notNull(),
  data: text("data"),
  inTrash: text("in_trash"),
  logo: text("logo"),
  bannerUrl: text("banner_url"),
});

export const folders = pgTable("folders", {
  id: uuid("id").defaultRandom().primaryKey().notNull(),
  createdAt: timestamp("created_at", {
    withTimezone: true,
    mode: "string",
  })
    .defaultNow()
    .notNull(),
  title: text("title").notNull(),
  iconId: text("icon_id").notNull(),
  data: text("data"),
  inTrash: text("in_trash"),
  bannerUrl: text("banner_url"),
  workspaceId: uuid("workspace_id")
    .notNull()
    .references(() => workspaces.id, {
      onDelete: "cascade",
    }),
});

export const files = pgTable("files", {
  id: uuid("id").defaultRandom().primaryKey().notNull(),
  createdAt: timestamp("created_at", {
    withTimezone: true,
    mode: "string",
  })
    .defaultNow()
    .notNull(),
  title: text("title").notNull(),
  iconId: text("icon_id").notNull(),
  data: text("data"),
  inTrash: text("in_trash"),
  bannerUrl: text("banner_url"),
  workspaceId: uuid("workspace_id")
    .notNull()
    .references(() => workspaces.id, {
      onDelete: "cascade",
    }),
  folderId: uuid("folder_id")
    .notNull()
    .references(() => folders.id, {
      onDelete: "cascade",
    }),
});

// export const subscriptions = pgTable("subscriptions", {
//   id: text("id").primaryKey().notNull(),
//   userId: uuid("user_id").notNull(),
//   status: subscriptionStatus("status"),
//   metadata: jsonb("metadata"),
//   priceId: text("price_id").references(() => prices.id),
//   quantity: integer("quantity"),
//   cancelAtPeriodEnd: boolean("cancel_at_period_end"),
//   created: timestamp("created", { withTimezone: true, mode: "string" })
//     .default(sql`now()`)
//     .notNull(),
//   currentPeriodStart: timestamp("current_period_start", {
//     withTimezone: true,
//     mode: "string",
//   })
//     .default(sql`now()`)
//     .notNull(),
//   currentPeriodEnd: timestamp("current_period_end", {
//     withTimezone: true,
//     mode: "string",
//   })
//     .default(sql`now()`)
//     .notNull(),
//   endedAt: timestamp("ended_at", {
//     withTimezone: true,
//     mode: "string",
//   }).default(sql`now()`),
//   cancelAt: timestamp("cancel_at", {
//     withTimezone: true,
//     mode: "string",
//   }).default(sql`now()`),
//   canceledAt: timestamp("canceled_at", {
//     withTimezone: true,
//     mode: "string",
//   }).default(sql`now()`),
//   trialStart: timestamp("trial_start", {
//     withTimezone: true,
//     mode: "string",
//   }).default(sql`now()`),
//   trialEnd: timestamp("trial_end", {
//     withTimezone: true,
//     mode: "string",
//   }).default(sql`now()`),
// });

// export const collaborators = pgTable("collaborators", {
//   id: uuid("id").defaultRandom().primaryKey().notNull(),
//   workspaceId: uuid("workspace_id")
//     .notNull()
//     .references(() => workspaces.id, { onDelete: "cascade" }),
//   createdAt: timestamp("created_at", {
//     withTimezone: true,
//     mode: "string",
//   })
//     .defaultNow()
//     .notNull(),
//   userId: uuid("user_id")
//     .notNull()
//     .references(() => users.id, { onDelete: "cascade" }),
// });

// //Dont Delete!!!
// export const productsRelations = relations(products, ({ many }) => ({
//   prices: many(prices),
// }));

// export const pricesRelations = relations(prices, ({ one }) => ({
//   product: one(products, {
//     fields: [prices.productId],
//     references: [products.id],
//   }),
// }));
